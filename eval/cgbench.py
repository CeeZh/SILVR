# Reference: https://github.com/CG-Bench/CG-Bench/tree/main/run

import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from collections import defaultdict
import numpy as np
import yaml
import pandas as pd
# from utils import save_json, load_json, save_pkl, load_pkl, makedir
import json
import ast


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def extract_characters_regex(response, all_choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return ""

    response = response.replace("\n", "")

    # Step 1: Clean up punctuation from the response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match
    # print(response)

    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True
    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
    # Step 4: If no candidates, look for choices with a space after (A B C D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    # # Step 5: If no candidates and response has more than 5 tokens, try parsing based on content
    # if len(candidates) == 0 and len(response.split()) > 5:
    #     for index, ans in index2ans.items():
    #         if ans.lower() in response.lower():
    #             candidates.append(index)
    #             index_ans = False  # It's content answer, not an index

    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = ""

    # Step 7: If multiple candidates found, use the one appearing last
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_period:
            for can in candidates:
                index = response.rfind(f"{can}.")
                start_indexes.append(index)
        elif ans_with_colon:
            for can in candidates:
                index = response.rfind(f"{can}:")
                start_indexes.append(index)
        elif ans_with_brack:
            for can in candidates:
                index = response.rfind(f"({can})")
                start_indexes.append(index)
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # Get the last one (max index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def eval_cgbench(folder_path, qa_data):
    categories = list(qa_data['sub_category'].unique())
    results = {key: {'num_corrects':0, 'num_total':0} for key in categories}
    num_corrects, num_total = 0, 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        # if len(example['prompt'][0]) <= 2*len("This video's subtitles are listed below:\n\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\nIn the video, when Latin dancer No. 15 came on stage, this group of men had a number tag attached to their backs. What was the number on the number tag?\n\nOptions.\nA: 23\nB: 39\nC: 11\nD: 13\nE: 31\nF: 21\nG: 32\n\n\nThe answer is:"):
        #     continue
        example_anno = qa_data.loc[vid_idx]
        results[example_anno['sub_category']]['num_total'] += 1
        if extract_characters_regex(example['response']) == example_anno['right_answer']:
            results[example_anno['sub_category']]['num_corrects'] += 1
            num_corrects += 1
        num_total += 1
    for cat in results:
        if results[cat]['num_total'] > 0:
            results[cat]['accuracy'] = f"{results[cat]['num_corrects'] / results[cat]['num_total']*100: .1f}"
        else:
            results[cat]['accuracy'] = "0"
    results['Average'] = {
        'num_corrects': num_corrects,
        'num_total': num_total,
        'accuracy': f"{num_corrects / num_total*100: .1f}" if num_total > 0 else "0"
    }
    return results



def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list.
    Assumes each interval is a list [start, end].
    """
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last_merged = merged[-1]

        # Check if there is an overlap
        if current[0] <= last_merged[1]:
            # Merge the current interval with the last one
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)

    return merged

def calculate_intervals_iou(intervals1, intervals2):
    """
    Calculate the IoU of two lists of intervals.
    Each list contains intervals represented as [start, end].
    """
    # Merge overlapping intervals in both lists
    merged1 = merge_intervals(intervals1)
    merged2 = merge_intervals(intervals2)

    # Calculate total length of intervals for both lists
    def total_length(merged_intervals):
        return sum(end - start for start, end in merged_intervals)

    length1 = total_length(merged1)
    length2 = total_length(merged2)

    # Calculate intersection length
    intersection_length = 0
    for interval1 in merged1:
        for interval2 in merged2:
            intersection_start = max(interval1[0], interval2[0])
            intersection_end = min(interval1[1], interval2[1])
            intersection_length += max(0, intersection_end - intersection_start)
    # Calculate union length
    union_length = length1 + length2 - intersection_length
    # IoU is intersection divided by union
    iou = intersection_length / union_length if union_length > 0 else 0
    return iou


def eval_cgbench_miou(folder_path, qa_data):
    iou_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data.loc[vid_idx]
        try:
            parsed = ast.literal_eval(example['response'])
        except (SyntaxError, ValueError) as e:
            # print(f"Error: {e}. Invalid input for file {filename} with value {parsed}. Set to [[0, 1]]")
            # parsed = [[0, 1]]
            pass
        iou = calculate_intervals_iou(example_anno['clue_intervals'], parsed)
        iou_list.append(iou)
    results = {
        'miou': sum(iou_list) / len(iou_list) * 100,
        'num_evaluated': len(iou_list),
        'num_total': len(qa_data)
    }
    return results