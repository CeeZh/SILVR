import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml

from utils import save_json, load_json, save_pkl, load_pkl, makedir



def parse_multi_choice_response(response, all_choices=['A', 'B', 'C', 'D', 'E']):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return ""

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


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]


def generate_submission_hourvideo(folder_path, qa_data, submission_reference_path, output_path):
    submission_reference = load_json(submission_reference_path)
    num_valid, num_answered, num_total = 0, 0, 0
    for k, v in submission_reference.items():
        num_total += len(v["benchmark_dataset"])

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        response = example['response']
        pred_ans = parse_multi_choice_response(response)
        # print(pred_ans)
        example_anno = qa_data.loc[vid_idx]

        video_uid, qid = example_anno["video_uid"], example_anno['qid']

        for i in range(len(submission_reference[video_uid]['benchmark_dataset'])):
            if qid == submission_reference[video_uid]['benchmark_dataset'][i]["qid"]:
                if pred_ans in ['A', 'B', 'C', 'D', 'E']:
                    num_valid += 1
                    submission_reference[video_uid]['benchmark_dataset'][i]["predicted_answer_label"] = pred_ans
                num_answered += 1

    save_json(submission_reference, output_path)
    print(f"num_missed: {num_total-num_answered}, num_answered: {num_answered}, num_total: {num_total}, num_valid: {num_valid}")