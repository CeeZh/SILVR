# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmworld/utils.py

import datetime
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml

from utils import save_json, load_json, save_pkl, load_pkl, makedir


DISCIPLINES = ["Tech & Engineering", "Science", "Health & Medicine", "Sports & Arts", "Game", "Business", "Embodied Tasks"]


def extract_characters_regex(s):
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


def mmworld_process_results(doc, results):
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

    discipline = doc["discipline"]
    data_dict = {"video_id": doc["video_id"], "discipline": discipline, "pred_answer": pred_ans, "answer": doc["correct_answer_label"].upper()}

    return {f"mmworld_accuracy": data_dict}


def mmworld_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    eval_results = {}
    category2score = {}

    for category in DISCIPLINES:
        key = f"{category}"
        category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["discipline"]
        key = f"{category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for category in DISCIPLINES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        print(f"Evaluation on DISCIPLINES: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        eval_results[category] = 100 * total_correct / total_answered if total_answered > 0 else 0

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    print(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    eval_results["Overall"] = 100 * total_correct / total_answered if total_answered > 0 else 0
    return 100 * total_correct / total_answered if total_answered > 0 else 0, eval_results


def eval_mmworld(folder_path, qa_data):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data.loc[vid_idx]
        results.append(mmworld_process_results(example_anno, [example['response']])['mmworld_accuracy'])
    
    _, printable_results = mmworld_aggregate_results(results)
    return printable_results
