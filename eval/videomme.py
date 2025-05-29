# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/videomme/utils.py

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


VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

replace_prompt = " Please answer yes or no."


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


def extract_characters_regex2(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def videomme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {f"videomme_perception_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    eval_results = defaultdict(dict)

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        eval_results['Category'][category] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        eval_results['Sub Category'][sub_cate] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        eval_results['Task Category'][task_cate] = f"{task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        print(f"Evaluation on video Type Count: {video_type}: {total_answered}")
        eval_results['Video Type'][video_type] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        eval_results['Video Type Count'][video_type] = f"{total_answered}"

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    print(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    eval_results['overall'] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
    return 100 * total_correct / total_answered if total_answered > 0 else 0, category2score, eval_results


def eval_videomme(folder_path, qa_data):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data.loc[vid_idx]
        results.append(videomme_process_results(example_anno, [example['response']])['videomme_perception_score'])
    
    overall_acc, category2score, eval_results = videomme_aggregate_results(results)
    return eval_results
