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


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def extract_characters_regex(response, all_choices=['A', 'B', 'C', 'D']):
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


def eval_egolife(folder_path, qa_data):
    categories = ['EntityLog', 'EventRecall', 'HabitInsight', 'RelationMap', 'TaskMaster']
    results = {key: {'num_corrects':0, 'num_total':0} for key in categories}
    num_corrects, num_total = 0, 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data[vid_idx]
        results[example_anno['type']]['num_total'] += 1
        if extract_characters_regex(example['response']) == example_anno['answer']:
            results[example_anno['type']]['num_corrects'] += 1
            num_corrects += 1
        num_total += 1    
    accuracy_list = []
    for cat in results:
        if results[cat]['num_total'] > 0:
            results[cat]['accuracy'] = f"{results[cat]['num_corrects'] / results[cat]['num_total'] * 100:.1f}"
        else:
            results[cat]['accuracy'] = "0"
        accuracy_list.append(float(results[cat]['accuracy']))
    results['Average'] = {
        'num_corrects': num_corrects,
        'num_total': num_total,
        'accuracy': f"{num_corrects / num_total * 100:.1f}" if num_total > 0 else "0"
    }
    results['Average_class_mean'] =  f"{sum(accuracy_list)/len(accuracy_list):.1f}" if len(accuracy_list) > 0 else "0"
    return results
