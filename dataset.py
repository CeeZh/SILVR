from utils import save_json, load_json, save_pkl, load_pkl, makedir
from torch.utils.data import Dataset
import pandas as pd
import pdb
from pprint import pprint
import os
import re
from datetime import timedelta
import json
from pathlib import Path
  

def format_seconds(seconds):
    return str(timedelta(seconds=seconds)).rjust(8, '0')

def extract_subtitles_srt(subtitle_dir, video_id, stride=1, include_time=True):
    subtitle_file = os.path.join(subtitle_dir, f'{video_id}.srt')
    
    if not os.path.exists(subtitle_file):
        print(f"Subtitle file not found for videoID: {video_id} and path {subtitle_file}")
        return ""
    
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove timestamps and numbers; extract only subtitle text
    subtitles = []
    cur_num = 0
    if include_time:
        for line in lines:
            line = line.strip()
            if '<' in line and '>' in line:
                line = re.sub(r'<.*?>', '', line)  # Remove HTML tags
            if cur_num % stride == 0:
                subtitles.append(line)
            if len(line) == 0:
                cur_num += 1
    else:
        for line in lines:
            line = line.strip()
            if not re.match(r'^[0-9]+$', line) and '-->' not in line:  # Skip numbers and timestamps
                line = re.sub(r'<.*?>', '', line)  # Remove HTML tags
                if cur_num % stride == 0 and line: 
                    subtitles.append(line)
                cur_num += 1
    subtitles = subtitles[::stride] 
    
    return '\n'.join(subtitles)

def extract_subtitles_json(subtitle_dir, subtitle_fn, include_time=True):
    subtitle_file = os.path.join(subtitle_dir, subtitle_fn)
    
    if not os.path.exists(subtitle_file):
        print(f"Subtitle file not found for video: {subtitle_fn} and path {subtitle_file}")
        return ""
    
    subtitle_elements = load_json(subtitle_file)

    if include_time:
        subtitles_list = []
        for ele in subtitle_elements:
            if 'start' in ele:
                subtitles_list.append(f"{ele['start']} --> {ele['end']}\n{ele['line']}\n")
            elif 'timestamp' in ele:
                subtitles_list.append(f"{ele['timestamp'][0]} --> {ele['timestamp'][1]}\n{ele['text']}\n")
        subtitles = '\n'.join(subtitles_list)
    else:
        subtitles_list = [ele['line'] for ele in subtitle_elements]
        subtitles = ' '.join(subtitles_list)

    return subtitles


def extract_subtitles_cgbench(subtitle_dir, video_id, include_time=True, format_time=False):
    subtitle_file = os.path.join(subtitle_dir, f'{video_id}.srt')
    
    if not os.path.exists(subtitle_file):
        print(f"Subtitle file not found for videoID: {video_id} and path {subtitle_file}")
        return ""
    
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    def time_to_seconds(time_str):
        """Convert time format 'HH:MM:SS,ms' to seconds (float)."""
        # print(time_str)
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s)

    subtitles = []
    for i, line in enumerate(lines):
        if '-->' in line:
            if include_time:
                if format_time:
                    subtitles.append(lines[i])
                else:
                    time_start = time_to_seconds(line.split('-->')[0].strip())
                    time_end = time_to_seconds(line.split('-->')[1].strip())
                    subtitles.append(f"{time_start} --> {time_end}\n")
                subtitles.append(lines[i+1])
                subtitles.append('\n')
            else:
                subtitles.append(lines[i+1])
    return ''.join(subtitles)


def extract_subtitles_egolife(subtitle_dir, query_date, query_time, include_time=True, context_window=300000):

    def extract_subtitles_egolife_one(subtitle_dir, query_time, include_time=include_time, context_window=context_window):
        if not os.path.exists(subtitle_dir):
            print(f"Subtitle file not found for path {subtitle_dir}")
            return ""
        
        with open(subtitle_dir, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        start_time = int(f'{query_time[:2]}000000')
        # Remove timestamps and numbers; extract only subtitle text
        subtitles = []
        for i, line in enumerate(lines):
            line = line.strip()
            if '-->' in line:
                time_str_start = line.split('-->')[0].strip().replace(':', '').replace(',', '')[:-1]
                time_str_end = line.split('-->')[1].strip().replace(':', '').replace(',', '')[:-1]
                if start_time + int(time_str_start) + context_window >= int(query_time) and start_time + int(time_str_end) <= int(query_time):
                    if include_time:
                        time_start = line.split('-->')[0].strip()
                        time_start = query_time[:2] + time_start[2:]
                        time_end = line.split('-->')[1].strip()
                        time_end = query_time[:2] + time_end[2:]
                        subtitles.append(f"{time_start} --> {time_end}\n")
                    subtitles.append(lines[i+2])
                    if include_time:
                        subtitles.append('\n')
        return ''.join(subtitles)

    cur_subtitle_filename = f'A1_JAKE_{query_date}_{query_time[:2]}000000.srt'
    cur_subtitle_dir = os.path.join(subtitle_dir, query_date, cur_subtitle_filename)
    cur_query_time = query_time
    cur_context_window = context_window
    subtitle = extract_subtitles_egolife_one(cur_subtitle_dir, cur_query_time, include_time=include_time, context_window=cur_context_window)

    relative_query_time = int(query_time[2:])
    if relative_query_time < context_window:
        subtitle_filenames = os.listdir(os.path.join(subtitle_dir, query_date))
        subtitle_filenames.sort()
        if cur_subtitle_filename in subtitle_filenames:
            prev_subtitle_filename_index = subtitle_filenames.index(cur_subtitle_filename) - 1
        else:
            prev_subtitle_filename_index = -1
        if prev_subtitle_filename_index >= 0:
            prev_subtitle_filename = subtitle_filenames[prev_subtitle_filename_index]
            if int(cur_subtitle_filename[-12:-10]) - int(prev_subtitle_filename[-12:-10]) == 1:
                prev_subtitle_dir = os.path.join(subtitle_dir, query_date, prev_subtitle_filename)
                prev_query_time = prev_subtitle_filename[-12:-10] + '6' + prev_subtitle_filename[-9:-4]
                prev_context_window = context_window - relative_query_time
                prev_subtitle = extract_subtitles_egolife_one(prev_subtitle_dir, prev_query_time, include_time=include_time, context_window=prev_context_window)
                subtitle = prev_subtitle + subtitle
    return subtitle


def extract_captions(caption_dir, video_id, stride, clip_length, include_time=True, format_time=True):
    caption_file_path = os.path.join(caption_dir, f'{video_id}.txt')

    if not os.path.exists(caption_file_path):
        print(f"Caption file not found for video_id: {video_id} and path {caption_file_path}")
        return ""
    
    with open(caption_file_path, "r", encoding="utf-8") as file:
        vid_captions_list = file.readlines()  # Read the entire file as a single string
    vid_captions_list = vid_captions_list[::stride]

    if not include_time:
        return ''.join(vid_captions_list)

    timestamped_captions = ""
    for i in range(len(vid_captions_list)):
        start_time, end_time = i*stride*clip_length, i*stride*clip_length+clip_length
        if format_time:
            start_time, end_time = format_seconds(start_time), format_seconds(end_time)
        timestamped_captions += f"{start_time} --> {end_time}\n{vid_captions_list[i]}"
    return timestamped_captions

def extract_captions_mmworld(caption_dir, video_id, stride, clip_length, include_time=True):
    caption_file_path_candidates = [
        os.path.join(caption_dir, f'{video_id}.txt'),
        os.path.join(caption_dir, f'shorts:{video_id}.txt'),
        os.path.join(caption_dir, f"{Path(video_id).parent}/shorts:{Path(video_id).name}.txt")
    ]
    caption_file_path = ''
    for el in caption_file_path_candidates:
        if os.path.exists(el):
            caption_file_path = el
            break
    if len(caption_file_path) == 0:
        print(f"Caption file not found for video_id: {video_id}")
        return ""
    
    with open(caption_file_path, "r", encoding="utf-8") as file:
        vid_captions_list = file.readlines()  # Read the entire file as a single string
    vid_captions_list = vid_captions_list[::stride]

    if not include_time:
        return ''.join(vid_captions_list)

    timestamped_captions = ""
    for i in range(len(vid_captions_list)):
        start_time, end_time = format_seconds(i*stride*clip_length), format_seconds(i*stride*clip_length+clip_length)
        timestamped_captions += f"{start_time} --> {end_time}\n{vid_captions_list[i]}"
    return timestamped_captions


def extract_captions_hourvideo(caption_dir, video_id, caption_keys, stride, clip_length, include_time=True):
    # keys = ['Scene Context', 'Motion Description', 'Spatial Relationship Analysis', 'Detailed Object Analysis', 'Temporal Relationship Context', 'Additional Details', 'Summary']

    caption_folder_path = os.path.join(caption_dir, video_id)

    if not os.path.exists(caption_folder_path):
        print(f"Caption folder not found for video_id: {video_id} and path {caption_folder_path}")
        return ""

    fn_list = os.listdir(caption_folder_path)
    fn_list.sort(key=lambda x: int(x.split('.')[0]))  # Sort by the number before the dot

    vid_captions_list = []
    for fn in fn_list:
        with open(os.path.join(caption_folder_path, fn), 'r') as f:
            caption_json = f.readlines()
        caption_json = ''.join(caption_json[1:-1])
        caption_json = json.loads(caption_json)
        parts = []
        for key in caption_keys:
            parts.append(f"{key}.\n{caption_json[key]}")
        caption = '\n'.join(parts)
        vid_captions_list.append(caption)
    vid_captions_list = vid_captions_list[::stride]

    if not include_time:
        return ''.join(vid_captions_list)

    timestamped_captions_list = []
    for i in range(len(vid_captions_list)):
        start_time, end_time = format_seconds(i*stride*clip_length), format_seconds(i*stride*clip_length+clip_length)
        timestamped_captions_list.append(f"{start_time} --> {end_time}\n{vid_captions_list[i]}")
    return "\n".join(timestamped_captions_list)

def extract_audio_captions(caption_dir, video_id):
    caption_file_path = os.path.join(caption_dir, f'{video_id}.txt')

    if not os.path.exists(caption_file_path):
        print(f"Audio caption file not found for video_id: {video_id}")
        return ""
    
    with open(caption_file_path, "r", encoding="utf-8") as file:
        vid_captions_list = file.readlines()  # Read the entire file as a single string

    return ''.join(vid_captions_list)


def extract_captions_egolife(caption_dir, query_date, query_time, include_time=True, context_window=300000):
    caption_dir = os.path.join(caption_dir, query_date)

    def get_files_in_range(folder_path, target_timestamp, range_offset=300000):
        start = target_timestamp - range_offset
        end = target_timestamp
        matched_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                ts = int(filename.split('_')[-1].split('.')[0])
                if start <= ts <= end:
                    matched_files.append(filename)
        return sorted(matched_files)

    vid_captions_list = []
    filenames = get_files_in_range(caption_dir, int(query_time), min(context_window, int(query_time[2:])))
    if int(query_time[2:]) < context_window:
        query_time_prev = f"{(int(query_time[:2])-1)}6{'0'*(len(query_time[2:])-1)}"
        filenames_prev = get_files_in_range(caption_dir, int(query_time_prev), context_window - int(query_time[2:]))
        filenames = filenames_prev + filenames
    for i, filename in enumerate(filenames):
        with open(os.path.join(caption_dir, filename), "r", encoding="utf-8") as file:
            captions_in_this_file = file.readlines()  # Read the entire file as a single string
            captions_in_this_file = [el for el in captions_in_this_file if el != '\n']
        if not include_time:
            vid_captions_list.extend(captions_in_this_file)
        else:
            ts_cur = filename.split('_')[-1].split('.')[0]
            if i < len(filenames) - 1:
                ts_next = filenames[i+1].split('_')[-1].split('.')[0]
                ts_cur_to_show = f"{ts_cur[:2]}:{ts_cur[2:4]}:{ts_cur[4:6]}.{ts_cur[6:]}"
                ts_next_to_show = f"{ts_next[:2]}:{ts_next[2:4]}:{ts_next[4:6]}.{ts_next[6:]}"
                vid_captions_list.append(f"{ts_cur_to_show} --> {ts_next_to_show}\n")
                vid_captions_list.extend(captions_in_this_file)
                vid_captions_list.append('\n')
    return ''.join(vid_captions_list)
    
    

class BaseDataset(Dataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        '''
        num_examples_to_run < 0: run all
        '''
        self.args = args
        self.anno = self.load_anno(args.anno_path)

        to_exclude = set(to_exclude)
        example_ids = []
        for i in range(len(self.anno)):
            if i in to_exclude:
                continue
            example_ids.append(i)
        
        if num_examples_to_run >= 0:
            example_ids = example_ids[:num_examples_to_run]
        self.example_ids = example_ids

        # print(len(self.example_ids))
        # pdb.set_trace()

    def load_anno(self, anno_path):
        return pd.read_parquet(anno_path)
    
    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, idx):
        pass


class VideoMMEDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['videoID'][idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        if len(self.args.audio_caption_path) > 0:
            audio_captions = extract_audio_captions(self.args.audio_caption_path, video_id)
        else:
            audio_captions = ""
        options = list(self.anno["options"][idx])
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': options,
            'duration': self.anno["duration"][idx],
            'sub_category': self.anno["sub_category"][idx],
            'domain': self.anno["domain"][idx],
            'task_type': self.anno["task_type"][idx],
            'answer': self.anno["answer"][idx],
            'subtitle': subtitle,
            'caption': caption,
            'audio_captions': audio_captions
        }
        return data


class VideoMMMUDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['id'][idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        options = list(self.anno["options"][idx])

        question_caption_path = os.path.join(self.args.image_caption_path, f"{self.anno['id'][idx]}.txt")
        if not os.path.exists(question_caption_path):
            question_caption = ""
            # print(f"No question caption for video {self.anno['id'][idx]}.")
        else:
            with open(question_caption_path, 'r') as f:
                question_caption = f.read()

        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'question_caption': question_caption,
            'options': options,
            'question_type': self.anno["question_type"][idx],
            'answer': self.anno["answer"][idx],
            'subtitle': subtitle,
            'caption': caption
        }
        return data


class LongVideoBenchDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['video_id'][idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_json(self.args.subtitle_path, self.anno['subtitle_path'][idx], include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        options = [self.anno['option1'][idx], self.anno['option2'][idx], self.anno['option3'][idx], self.anno['option4'][idx]]
        # for op in options:
        #     print(op)
        #     print(type(op))
        #     pdb.set_trace()
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': options,
            'duration': float(self.anno["duration"][idx]),
            'duration_group': int(self.anno["duration_group"][idx]),
            'answer': int(self.anno["correct_choice"][idx]),
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data

class CinePileDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['videoID'][idx]

        subtitle_time = not self.args.subtitle_no_time
        subtitle_raw = self.anno['subtitles'][idx]
        subtitle = "\n".join(line.replace("<subtitle> ", "") for line in subtitle_raw.strip().split("\n"))

        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': self.anno['choices'][idx].tolist(),
            'question_category': self.anno['question_category'][idx],
            'hard_split': self.anno['hard_split'][idx],
            'answer_key': self.anno["answer_key"][idx],
            'answer': int(self.anno["answer_key_position"][idx]),
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data
    

class MLVUDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['video_name'][idx].split('.')[0]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        options = self.anno['candidates'][idx]
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': list(options),
            'duration': float(self.anno["duration"][idx]),
            'task_type': self.anno["task_type"][idx],
            'answer': self.anno["answer"][idx],
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data


class MMVUDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def load_anno(self, anno_path):
        return pd.read_json(anno_path)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['id'][idx]
        # subtitle_id = self.anno['youtube_url'][idx].split('=')[1].split('&')[0]
        youtube_id = self.anno['youtube_url'][idx]
        if '=' in youtube_id:
            subtitle_id = youtube_id.split('=')[1].split('&')[0]
        else:
            subtitle_id = youtube_id.split('/')[-1] 
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, subtitle_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_id = self.anno['video'][idx].split("videos/")[-1].split(".")[0]
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, caption_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        options_dict = self.anno['choices'][idx]
        if len(options_dict['A']) == 0:
            options = []
        else:
            options = [options_dict['A'], options_dict['B'], options_dict['C'], options_dict['D'], options_dict['E']]
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'id': video_id,  # used in evaluation
            'question': self.anno['question'][idx],
            'options': options,
            'choices': options_dict,
            'question_type': self.anno["question_type"][idx],
            'metadata': self.anno["metadata"][idx],
            'answer': self.anno["answer"][idx],
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data


class MMWorldDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['video_id'][idx]
        video_id_for_subtitle = video_id.split('/')[-1].split('.')[0]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id_for_subtitle, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        video_id_for_caption = f"{video_id.split('.')[0]}/{video_id.split('/')[-1].split('.')[0]}"
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions_mmworld(self.args.caption_path, video_id_for_caption, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""

        options_dict = self.anno['options'][idx]
        options = [options_dict['a'], options_dict['b'], options_dict['c'], options_dict['d']]

        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': options,
            'requires_audio': bool(self.anno['requires_audio'][idx]),
            'discipline': self.anno["discipline"][idx],
            'answer': self.anno["correct_answer_label"][idx],
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data


class HourVideoDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['video_uid'][idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions_hourvideo(self.args.caption_path, video_id, self.args.caption_keys, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""

        options = [self.anno['answer_1'][idx], self.anno['answer_2'][idx], self.anno['answer_3'][idx], self.anno['answer_4'][idx], self.anno['answer_5'][idx]]  # TODO: add captions

        option_captions = []
        if '.png' in options[0]:
            for option in options:
                option = option.replace('.png', '.txt')
                option_caption_path = os.path.join(self.args.hourvideo_image_caption_path, option)
                if os.path.exists(option_caption_path):
                    with open(option_caption_path, 'r') as f:
                        option_caption = f.read()
                        option_captions.append(option_caption)
        if len(option_captions) > 0:
            options = option_captions

        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': options,
            'duration_in_seconds': float(self.anno['duration_in_seconds'][idx]),
            'task': self.anno["task"][idx],
            'scenarios': self.anno["scenarios"][idx],
            'answer': -1,
            'subtitle': subtitle,
            'caption': caption
        }
        
        return data


class VideoMMLUDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
        self.subtitles = self.load_subtitles(args.subtitle_path)

    def load_anno(self, anno_path):
        import json

        category_info = []
        with open(self.args.videommlu_category_file, 'r') as f:
            for line in f:
                category_info.append(json.loads(line))
        category_info = {el['video_id']: el['label'] for el in category_info}

        data = []
        with open(anno_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        examples = []
        for example in data:
            for i in range(len(example['reasoning_qa'])):
                examples.append(
                    {
                        'video_id': example['video_id'],
                        'question': example['reasoning_qa'][i]['question'],
                        'answer': example['reasoning_qa'][i]['answer'],
                        'type': category_info[example['video_id']]
                    }
                )
        return examples

    def load_subtitles(self, subtitle_dir):
        subtitles = {}
        if len(subtitle_dir) > 0:
            for element in load_json(subtitle_dir):
                subtitles[element['video_id']] = element['org_text']['transcript']
        return subtitles

    def extract_subtitles(self, video_id, include_time=True):    
        if video_id not in self.subtitles:
            print(f"Subtitle not found for video: {video_id}")
            return ""
            
        subtitles_raw = self.subtitles[video_id] if video_id in self.subtitles else []
        if include_time:
            subtitles_list = []
            for ele in subtitles_raw:
                subtitles_list.append(f"{format_seconds(ele['start'])} --> {format_seconds(ele['start']+ele['duration'])}\n{ele['text']}\n")
            subtitles = '\n'.join(subtitles_list)
        else:
            subtitles_list = [ele['text'] for ele in subtitles_raw]
            subtitles = '\n'.join(subtitles_list)
        return subtitles

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno[idx]['video_id']
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = self.extract_subtitles(video_id, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno[idx]['question'],
            'type': self.anno[idx]['type'],
            'answer': self.anno[idx]["answer"],
            'subtitle': subtitle,
            'caption': caption,
        }
        return data


class CGBenchDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def load_anno(self, anno_path):
        return pd.read_json(anno_path)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno['video_uid'][idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_cgbench(self.args.subtitle_path, video_id, include_time=subtitle_time, format_time=self.args.cgbench_task=='mc')
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time, format_time=self.args.cgbench_task=='mc')
        else:
            caption = ""
        options = list(self.anno["choices"][idx])
        data = {
            'global_idx': idx,
            'qid': int(self.anno['qid'][idx]),
            'video_id': video_id,
            'question': self.anno['question'][idx],
            'options': options,
            'duration': int(self.anno["duration"][idx]),
            'sub_category': self.anno["sub_category"][idx],
            'domain': self.anno["domain"][idx],
            'answer': self.anno["answer"][idx],
            'right_answer': self.anno["right_answer"][idx],
            'clue_intervals': [[float(i) for i in el] for el in self.anno['clue_intervals'][idx]],
            'subtitle': subtitle,
            'caption': caption
        }
        return data
    

class EgoLifeDataset(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def load_anno(self, anno_path):
        return load_json(anno_path)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_egolife(self.args.subtitle_path, self.anno[idx]['query_time']['date'], self.anno[idx]['query_time']['time'], context_window=self.args.egolife_context_window, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions_egolife(self.args.caption_path, self.anno[idx]['query_time']['date'], self.anno[idx]['query_time']['time'], context_window=self.args.egolife_context_window, include_time=caption_time)
        else:
            caption = ""
        options = [
            self.anno[idx]['choice_a'],
            self.anno[idx]['choice_b'],
            self.anno[idx]['choice_c'],
            self.anno[idx]['choice_d'],
        ]
        data = {
            'global_idx': idx,
            'qid': int(self.anno[idx]['ID']),
            'question': self.anno[idx]['question'],
            'options': options,
            'type': self.anno[idx]['type'],
            'query_time': self.anno[idx]['query_time'],
            'answer': self.anno[idx]["answer"],
            'subtitle': subtitle,
            'caption': caption
        }
        return data


class Minerva(BaseDataset):
    def __init__(self, args, to_exclude=[], num_examples_to_run=-1):
        super().__init__(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)

    def load_anno(self, anno_path):
        return load_json(anno_path)

    def __getitem__(self, idx):
        idx = self.example_ids[idx]
        video_id = self.anno[idx]['video_id']
        subtitle_time = not self.args.subtitle_no_time
        if len(self.args.subtitle_path) > 0:
            subtitle = extract_subtitles_srt(self.args.subtitle_path, video_id, stride=self.args.subtitle_stride, include_time=subtitle_time)
        else:
            subtitle = ""
        caption_time = not self.args.caption_no_time
        if len(self.args.caption_path) > 0:
            caption = extract_captions(self.args.caption_path, video_id, self.args.stride, self.args.clip_length, include_time=caption_time)
        else:
            caption = ""
        options = [
            self.anno[idx]['answer_choice_0'],
            self.anno[idx]['answer_choice_1'],
            self.anno[idx]['answer_choice_2'],
            self.anno[idx]['answer_choice_3'],
            self.anno[idx]['answer_choice_4'],
        ]
        answer_id = self.anno[idx]['answer_id']
        answer_char = chr(ord('A')+answer_id)
        data = {
            'global_idx': idx,
            'video_id': video_id,
            'question': self.anno[idx]['question'],
            'options': options,
            'question_type': self.anno[idx]['question_type'],
            'category': self.anno[idx]['category'],
            'answer': answer_char,
            'subtitle': subtitle,
            'caption': caption
        }
        return data


def get_dataset(args, to_exclude=None, num_examples_to_run=-1):
    if args.dataset.lower() == 'videomme':
        return VideoMMEDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif 'videommmu' in args.dataset.lower():
        return VideoMMMUDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'longvideobench':
        return LongVideoBenchDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'cinepile':
        return CinePileDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'mlvu':
        return MLVUDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'mmvu':
        return MMVUDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'mmworld':
        return MMWorldDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'hourvideo':
        return HourVideoDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'videommlu':
        return VideoMMLUDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'cgbench':
        return CGBenchDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'egolife':
        return EgoLifeDataset(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset.lower() == 'minerva':
        return Minerva(args, to_exclude=to_exclude, num_examples_to_run=num_examples_to_run)
    else:
        raise NotImplementedError()
