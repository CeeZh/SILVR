# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/longvideobench/utils.py

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image

from utils import save_json, load_json, save_pkl, load_pkl, makedir

def generate_submission_file(file_name, output_path, subpath="submissions"):
    path = os.path.join(output_path, subpath)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)
    return os.path.abspath(path)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
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
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    lvb_acc = {"id": id, "duration_group": int(doc["duration_group"]), "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }


def longvideobench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"], printable_results


def longvideobench_aggregate_results_for_submission(results, output_path):
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(output_path, "w") as f:
        json.dump(results_dict, f)
    print(f"Results saved to {output_path}.")


def eval_longvideobench(folder_path, qa_data):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data.loc[vid_idx]
        results.append(longvideobench_process_results(example_anno, [example['response']])['lvb_acc'])
    
    _, printable_results = longvideobench_aggregate_results(results)
    return printable_results

def generate_submission_longvideobench(folder_path, qa_data, output_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        example = load_json(os.path.join(folder_path, filename))
        example_anno = qa_data.loc[vid_idx]
        results.append(longvideobench_process_results(example_anno, [example['response']])['submission'])
    
    longvideobench_aggregate_results_for_submission(results, output_path)