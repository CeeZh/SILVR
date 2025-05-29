import os
from pathlib import Path
from utils import save_json, load_json, save_pkl, load_pkl, makedir
from eval import *
from dataset import *
from prompts import get_prompt
from model import get_model
from tqdm import tqdm
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import time
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--caption_path", default="", type=str) 
    parser.add_argument("--subtitle_path", default="", type=str)  
    parser.add_argument("--audio_caption_path", default="", type=str) 
    parser.add_argument("--anno_path", required=True, type=str)  
    parser.add_argument("--clip_length", default=64, type=int) 
    parser.add_argument("--stride", default=1, type=int) 
    parser.add_argument("--subtitle_stride", default=1, type=int) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)

    # prompt
    parser.add_argument("--prompt_type", default="videomme", type=str)
    parser.add_argument("--caption_no_time", action='store_true')
    parser.add_argument("--subtitle_no_time", action='store_true')
    parser.add_argument("--force_caption", action='store_true')
    parser.add_argument("--force_subtitle", action='store_true')

    # output
    parser.add_argument("--output_base_path", required=True, type=str)  

    # model
    parser.add_argument("--model", default="deepseek-reasoner", type=str)
    parser.add_argument("--endpoint", default="", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--api_url", default="https://api.deepseek.com/v1/chat/completions", type=str)
    parser.add_argument("--openai_api_key", default="", type=str)

    
    # videommmu
    parser.add_argument("--image_caption_path", default="", type=str)

    # hourvideo
    parser.add_argument("--hourvideo_image_caption_path", default="", type=str)
    parser.add_argument("--caption_keys", nargs='+', default=['Scene Context', 'Motion Description', 'Spatial Relationship Analysis', 'Detailed Object Analysis', 'Temporal Relationship Context', 'Additional Details', 'Summary'])
    parser.add_argument("--submission_reference_path", default="", type=str)

    # egolife
    parser.add_argument("--egolife_context_window", default=300000, type=int)  # 30min

    # cgbench
    parser.add_argument("--cgbench_task", default='mc', type=str)   # mc, miou

    # videommlu
    parser.add_argument("--videommlu_category_file", default='data/videommlu/Video-MMLU/video_sources.jsonl', type=str)

    # eval
    parser.add_argument("--backup_path", default="", type=str)

    # other
    parser.add_argument("--hf_token", default="", type=str)
    parser.add_argument("--single_process", action='store_true')
    parser.add_argument("--num_workers", default=64, type=int) 
    parser.add_argument("--time_sleep", default=0, type=float) 
    parser.add_argument("--disable_infer", action='store_true')
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--from_scratch", action='store_true')

    return parser.parse_args()


def eval_dataset(args, output_path, dataset, submission_file_path=None):
    if args.dataset.lower() == 'videomme':
        results = eval_videomme(output_path, dataset.anno)
    elif 'videommmu' in args.dataset.lower():
        results = eval_videommmu(output_path, dataset.anno)
    elif args.dataset.lower() == 'longvideobench':
        results = eval_longvideobench(output_path, dataset.anno)
        generate_submission_longvideobench(output_path, dataset.anno, submission_file_path)
    elif args.dataset.lower() == 'cinepile':
        results = eval_cinepile(output_path, dataset.anno)
    elif args.dataset.lower() == 'mlvu':
        results = eval_mlvu(output_path, dataset.anno)
    elif args.dataset.lower() == 'mmvu':
        results = eval_mmvu(output_path, dataset.anno)
    elif args.dataset.lower() == 'mmworld':
        results = eval_mmworld(output_path, dataset.anno)
    elif args.dataset.lower() == 'hourvideo':
        results = generate_submission_hourvideo(output_path, dataset.anno, args.submission_reference_path, submission_file_path)
    elif args.dataset.lower() == 'egolife':
        results = eval_egolife(output_path, dataset.anno)
    elif args.dataset.lower() == 'cgbench':
        if args.cgbench_task == 'mc':
            results = eval_cgbench(output_path, dataset.anno)
        elif args.cgbench_task == 'miou':
            results = eval_cgbench_miou(output_path, dataset.anno)
        else:
            raise NotImplementedError(f"The codebase does not support evaluation for task {args.cgbench_task}.")
    elif args.dataset.lower() == 'videommlu':
        results = eval_videommlu(output_path, args.anno_path, args.videommlu_category_file)
    elif args.dataset.lower() == 'minerva':
        results = eval_minerva(output_path, dataset.anno)
    else:
        raise NotImplementedError(f"The codebase does not support evaluation for {args.dataset}.")
    return results


def process_one(args, output_path, model, prompt_type, item, clip_length, time_sleep=1):
    if args.force_caption and len(item['caption']) == 0:
        return
    if args.force_subtitle and len(item['subtitle']) == 0:
        return
    prompt = get_prompt(prompt_type, item, clip_length)
    try:
        pred = model.forward("", prompt)
    except Exception as e:
        print(f"Error in Model. Error message: {e}")
        output_error_path = Path(output_path).parent / 'error'
        makedir(str(output_error_path))
        item['prompt'] = prompt
        save_json(item, os.path.join(output_error_path, f"{item['global_idx']}.json"))
        return
    output = {}
    output['prompt'] = prompt
    output.update(item)
    output.update(pred)
    if 'message' in output:
        del output['message']
    if 'subtitle' in output:
        del output['subtitle']
    if 'caption' in output:
        del output['caption']
    save_json(output, os.path.join(output_path, f"{item['global_idx']}.json"))
    
    # Sleep briefly between requests
    time.sleep(time_sleep)


def launch():
    args = parse_args()
    pprint(args)
    os.environ["HF_TOKEN"] = args.hf_token
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # output
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, 'logs')
    makedir(output_path)

    # save args
    save_json(vars(args), os.path.join(args.output_base_path, 'config.json'))

    # check processed questions
    processed_indices = []
    if not args.from_scratch:
        for filename in os.listdir(output_path):
            if filename.endswith('.json'):
                try:
                    vid_idx = int(filename.split('.')[0])
                except ValueError:
                    continue
                processed_indices.append(vid_idx)
                
    # get input
    dataset = get_dataset(args, to_exclude=processed_indices, num_examples_to_run=args.num_examples_to_run)

    if not args.disable_infer:
        # get model
        model = get_model(args)

        # answer
        if args.single_process:
            for item in tqdm(dataset):
                process_one(args, output_path, model, args.prompt_type, item, args.clip_length, time_sleep=args.time_sleep)
        else:
            max_inflight = args.num_workers + 30  # Adjust this as needed
            futures = []
            pbar = tqdm(total=len(dataset))
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                for i, item in enumerate(dataset):
                    pbar.update(1)
                    # print(f"{i} / {len(dataset)}")
                    # If too many tasks are in flight, wait for some to finish
                    while len(futures) >= max_inflight:
                        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        futures = list(not_done)

                    future = executor.submit(process_one, args, output_path, model, args.prompt_type, item, args.clip_length, time_sleep=args.time_sleep)
                    futures.append(future)
                # Wait for any remaining futures
                print('waiting for remaining features to complete')
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[Error] {e}")
            pbar.close()

    # eval
    if not args.disable_eval:
        submission_file_path = os.path.join(args.output_base_path, 'submission.json')
        results = eval_dataset(args, output_path, dataset, submission_file_path=submission_file_path)
        if results is not None:
            eval_output_path = os.path.join(args.output_base_path, 'results.json')
            save_json(results, eval_output_path)

        if len(args.backup_path) > 0 and os.path.exists(args.backup_path):
            output_path_with_backup = os.path.join(args.output_base_path, 'logs_with_backup')
            if os.path.exists(output_path_with_backup):
                shutil.rmtree(output_path_with_backup)
            shutil.copytree(output_path, output_path_with_backup, dirs_exist_ok=True)
            for fn in os.listdir(args.backup_path):
                try:
                    vid_idx = int(fn.split('.')[0])
                except ValueError:
                    continue
                src_path = os.path.join(args.backup_path, fn)
                tgt_path = os.path.join(output_path_with_backup, fn)
                if not os.path.exists(tgt_path):
                    shutil.copy(src_path, tgt_path)
            # eval
            submission_file_path = os.path.join(args.output_base_path, 'submission_with_backup.json')
            results = eval_dataset(args, output_path_with_backup, dataset, submission_file_path=submission_file_path)
            if results is not None:
                eval_output_path = os.path.join(args.output_base_path, 'results_with_backup.json')
                save_json(results, eval_output_path)


if __name__ == '__main__':
    launch()