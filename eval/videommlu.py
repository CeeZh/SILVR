# Reference: https://github.com/Espere-1119-Song/Video-MMLU/blob/main/post_eval/eval_reason_qa.py

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import ast
import pandas as pd
import os
import argparse
import numpy as np
from glob import glob
import random
import json
import argparse
import numpy as np
from collections import defaultdict
import json
import requests
import time
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def reason_qa(folder_path, save_path, source_file, category_file_path):    
    os.makedirs(save_path, exist_ok=True)
    category_info = []
    with open(category_file_path, 'r') as f:
        for line in f:
            category_info.append(json.loads(line))
    category_info = {el['video_id']: el['label'] for el in category_info}

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    backend_config = TurbomindEngineConfig(tp=4)
    gen_config = GenerationConfig(top_p=0.8,
                                  top_k=40,
                                  temperature=0,
                                  max_new_tokens=32)
    pipe = pipeline('Qwen/Qwen2.5-72B-Instruct',
                    backend_config=backend_config)

    qa_pairs = {}

    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_id = data['video_id']
            gt_qa_pairs = []
            for qa in data['captions_qa']:
                gt_question = qa['question'].replace("\\'", "'")
                gt_answer = qa['answer'].replace("\\'", "'")
                gt_qa_pairs.append({
                    'question': gt_question,
                    'answer': gt_answer
                })
            qa_pairs[video_id] = gt_qa_pairs

    pbar = tqdm(total=len(os.listdir(folder_path)), desc="Processing files")
    for filename in os.listdir(folder_path):
        pbar.update(1)
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        log = load_json(os.path.join(folder_path, filename))

        if os.path.exists(os.path.join(save_path, f"{vid_idx}.json")):
            continue

        global_idx = log['global_idx']
        video_id = log['video_id']
        pred = log['response']
        question = log['question']
        answer = log['answer']
        discipline = category_info[video_id]
    
        prompts = [[{
            'role': 'system',
            'content': 
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for reasoning-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer based on the following rules:"
                "------"
                "## INSTRUCTIONS:"
                "1. **Evaluate Reasoning Tasks Strictly:**"
                "   - The predicted answer must capture all critical concepts and details mentioned in the correct answer. "
                "   - If the correct answer mentions specific concepts or examples (e.g., 'odd numbers accumulate to form perfect squares'), the predicted answer must include these concepts or examples. "
                "   - Even if the phrasing differs, the key meaning and concepts must be preserved. However, omitting or altering key concepts or examples is **not acceptable**."
                "   - **Example 1:** If the correct answer is 'The construction method shows how odd numbers accumulate to form perfect squares,' the predicted answer must include 'odd numbers' and 'perfect squares.'"
                "   - **Example 2:** If the correct answer is 'To eliminate HBr and form an alkene,' the predicted answer must address the elimination of HBr as well."
                "   - Minor differences in phrasing are acceptable as long as the key information is retained."
                "   - **Critical Detail:** If any essential element (e.g., key terms, concepts, or examples) is missing from the predicted answer, the answer is considered incorrect."
                "   - Do **not** introduce new, unrelated information in the predicted answer."
        }], [{
            'role': 'user',
            'content': 
                "Please evaluate the following reasoning-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Ensure that the predicted answer captures all critical concepts and details from the correct answer, without omitting key elements. "
                "Minor rewording is acceptable, but the meaning and essential details must remain the same. "
                "If the predicted answer misses any critical concept or introduces unrelated information, it should be judged as incorrect. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'no', 'score': 3}."
        }]]
         
        response = pipe(prompts, gen_config=gen_config)
        try:
            judgement_string = response[-1].text
            judgement_dict = ast.literal_eval(judgement_string)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
            
        # Add the judgement_dict, video_id, question, answer, pred to the save file
        path = os.path.join(save_path, f"{global_idx}.json")
        save_json({
            'video_id': video_id, 
            'discipline': discipline,
            'judgement': judgement_dict, 
            'question': question, 
            'answer': answer, 
            'pred': pred
        }, path)


def reason_qa_api(folder_path, save_path, source_file, category_file_path):    
    category_info = []
    with open(category_file_path, 'r') as f:
        for line in f:
            category_info.append(json.loads(line))
    category_info = {el['video_id']: el['label'] for el in category_info}

    from together import Together

    # api_key = 'sk-qjfxjzmykvstnzqkuovaupqznslbdepmnyclfmcktrvtbpom'
    # model_name = 'Qwen/Qwen2.5-72B-Instruct'  
    api_key = '12c8d59c9580d78689a79942d3749b23ba2885009ec57bfc080449cd138e8f36'
    model_name = 'Qwen/Qwen2.5-72B-Instruct-Turbo'
    client = Together(api_key=api_key)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    gen_config = {
        'top_p': 0.8,
        'top_k': 40,
        'temperature': 0,
        'max_new_tokens': 32
    }

    qa_pairs = {}

    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_id = data['video_id']
            gt_qa_pairs = []
            for qa in data['captions_qa']:
                gt_question = qa['question'].replace("\\'", "'")
                gt_answer = qa['answer'].replace("\\'", "'")
                gt_qa_pairs.append({
                    'question': gt_question,
                    'answer': gt_answer
                })
            qa_pairs[video_id] = gt_qa_pairs
    pbar = tqdm(total=len(os.listdir(folder_path)), desc="Processing files")
    for filename in os.listdir(folder_path):
        pbar.update(1)
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        log = load_json(os.path.join(folder_path, filename))

        if os.path.exists(os.path.join(save_path, f"{vid_idx}.json")):
            continue

        global_idx = log['global_idx']
        video_id = log['video_id']
        pred = log['response']
        question = log['question']
        answer = log['answer']
        discipline = category_info[video_id]
    
        prompts = [{
            'role': 'system',
            'content': 
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for reasoning-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer based on the following rules:"
                "------"
                "## INSTRUCTIONS:"
                "1. **Evaluate Reasoning Tasks Strictly:**"
                "   - The predicted answer must capture all critical concepts and details mentioned in the correct answer. "
                "   - If the correct answer mentions specific concepts or examples (e.g., 'odd numbers accumulate to form perfect squares'), the predicted answer must include these concepts or examples. "
                "   - Even if the phrasing differs, the key meaning and concepts must be preserved. However, omitting or altering key concepts or examples is **not acceptable**."
                "   - **Example 1:** If the correct answer is 'The construction method shows how odd numbers accumulate to form perfect squares,' the predicted answer must include 'odd numbers' and 'perfect squares.'"
                "   - **Example 2:** If the correct answer is 'To eliminate HBr and form an alkene,' the predicted answer must address the elimination of HBr as well."
                "   - Minor differences in phrasing are acceptable as long as the key information is retained."
                "   - **Critical Detail:** If any essential element (e.g., key terms, concepts, or examples) is missing from the predicted answer, the answer is considered incorrect."
                "   - Do **not** introduce new, unrelated information in the predicted answer."
            }, 
            {
            'role': 'user',
            'content': 
                "Please evaluate the following reasoning-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Ensure that the predicted answer captures all critical concepts and details from the correct answer, without omitting key elements. "
                "Minor rewording is acceptable, but the meaning and essential details must remain the same. "
                "If the predicted answer misses any critical concept or introduces unrelated information, it should be judged as incorrect. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'no', 'score': 3}."
            }]
        # response = ""
        try:
            response = client.chat.completions.create(  
                model=model_name,  
                messages=prompts,  
                # temperature=0.7,  
                # max_tokens=1024,
                **gen_config
            )
        except Exception as e:
            print(f"Error in model. Response: {response}")
            time.sleep(5)
            continue
        response_text = response.choices[0].message.content
        # response_text = response["choices"][0]["message"]["content"]
        try:
            judgement_dict = ast.literal_eval(response_text)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
            
        # Add the judgement_dict, video_id, question, answer, pred to the save file
        path = os.path.join(save_path, f"{global_idx}.json")
        save_json({
            'video_id': video_id, 
            'discipline': discipline,
            'judgement': judgement_dict, 
            'question': question, 
            'answer': answer, 
            'pred': pred
        }, path)


def calculate_scores(results_path, output_file):
    # Initialize data structures to store scores and correct predictions
    discipline_scores = defaultdict(list)
    discipline_correct = defaultdict(int)
    discipline_counts = defaultdict(int)
    
    # Read the results file
    for filename in os.listdir(results_path):
        if filename.endswith('.json'):
            try:
                vid_idx = int(filename.split('.')[0])
            except ValueError:
                continue
        data = load_json(os.path.join(results_path, filename))
        discipline = data['discipline']
        score = data['judgement']['score']
        is_correct = data['judgement']['pred'].lower() == 'yes'
        
        # Add score to appropriate discipline
        discipline_scores[discipline].append(score)
        if is_correct:
            discipline_correct[discipline] += 1
        discipline_counts[discipline] += 1

    # Ensure all three disciplines are represented (even if zero samples)
    for discipline in ['Math', 'Physics', 'Chemistry']:
        if discipline not in discipline_counts:
            discipline_counts[discipline] = 0
            discipline_correct[discipline] = 0
            discipline_scores[discipline] = []
    
    # Calculate results for each discipline
    category_results = {}
    all_scores = []
    total_samples = 0
    total_correct = 0
    
    for discipline, count in sorted(discipline_counts.items()):
        correct = discipline_correct[discipline]
        accuracy = 0
        avg_score = 0
        
        if count > 0:
            accuracy = (correct / count) * 100
            avg_score = np.mean(discipline_scores[discipline])
            all_scores.extend(discipline_scores[discipline])
        
        category_results[discipline] = {
            "total_samples": count,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "average_score": avg_score
        }
        
        total_samples += count
        total_correct += correct
    
    # Calculate overall results
    overall_accuracy = 0
    overall_avg_score = 0
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        overall_avg_score = np.mean(all_scores) if all_scores else 0
    
    # Calculate final score (average of the three discipline scores)
    discipline_avg_scores = [category_results[d]["average_score"] for d in ["Math", "Physics", "Chemistry"]]
    final_score = sum(discipline_avg_scores) / 3

    discipline_acc_list = [category_results[d]["accuracy"] for d in ["Math", "Physics", "Chemistry"]]
    
    # Prepare the final results dictionary
    results = {
        "category_results": category_results,
        "overall_results": {
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "overall_average_score": overall_avg_score,
            "final_score": final_score,
            "class_mean": sum(discipline_acc_list) / len(discipline_acc_list)
        }
    }
    
    # Print and optionally save the results
    results_json = json.dumps(results, indent=4)
    print(results_json)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(results_json)
    
    # Print a simple summary
    print("\nSummary:")
    for discipline in sorted(category_results.keys()):
        print(f"{discipline} average score: {category_results[discipline]['average_score']:.4f}")
    print(f"Final score (average of three discipline scores): {final_score:.4f}")


def eval_main():
    source_file = '/mnt/arc/cezhang/projects/Video-MMLU/video_mmlu.jsonl'
    category_file_path = '/mnt/arc/cezhang/projects/Video-MMLU/video_sources.jsonl'
    # folder_path = '/mnt/arc/cezhang/projects/LLoVi/output/videommlu/subtitle+caption8/logs'
    folder_path = '/mnt/arc/cezhang/projects/LLoVi/output/videommlu/caption8/logs'
    reason_save_path = folder_path + '_eval'
    results_save_path = folder_path + '_eval.json'
    # reason_qa_api(folder_path, reason_save_path, source_file, category_file_path)
    # reason_qa(folder_path, reason_save_path, source_file, category_file_path)
    calculate_scores(reason_save_path, results_save_path)


def eval_videommlu(folder_path, anno_path, category_file_path):
    reason_save_path = folder_path + '_eval'
    results_save_path = folder_path + '_eval.json'
    reason_qa(folder_path, reason_save_path, anno_path, category_file_path)
    calculate_scores(reason_save_path, results_save_path)


if __name__ == "__main__":
    eval_main()


'''
python eval/videommlu.py

# Submit the job
sbatch \
  --cpus-per-task=32 \
  --gpus=4 \
  -p h100 \
  -o logs/%j.log \
  -e logs/%j.err \
  -J eval \
  --wrap="python eval/videommlu.py"
'''