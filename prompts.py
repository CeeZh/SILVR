from string import Template
import random


def get_prompt(prompt_type, data, clip_length):
    if prompt_type in {'standard', 'videomme', 'longvideobench', 'cinepile', 'mlvu', 'hourvideo', 'mmworld', 'egolife', 'cgbench', 'minerva'}:
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:" 
    elif prompt_type == 'standard_with_audio':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nThe video's audio captions are listed below:\n{data['audio_captions']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:" 
    elif prompt_type in {'open-ended', 'videommlu'}:
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nAnswer the following question with one sentence.\n\nQuestion.\n{data['question']}\n\nDo not miss any critical concept or introduce unrelated information to the question. The answer is:"
    elif prompt_type == 'videommmu':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        if '<image 1>' in data['question']:
            if data['question_caption'] == "":
                print(f"No question caption for video {data['video_id']}.")
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nAnswer the question based on all the provided information. If the options are given, please select the most accurate answer. In this case, please respond with only the letter (A, B, C, D, E, etc.) of the correct option. However, if the options are not given, please directly answer the question. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\nBelow is the caption of <image 1>.\n{data['question_caption']}\n\nOptions.\n{options_str}\n\nThe answer is:"
        else:
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'minerva_rubric':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"You will be given a question about a video and five possible answer options. This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\n Provide all steps required to come to the answer in your reasoning, and the following rubric will be used to judge the reasoning: (1) Perceptual correctness: was the relevant information perceived accurately from the video? (2) Temporal grounding: were time ranges provided for each piece of information from the video, and if so were they accurate? (3) Logical reasoning: was the reasoning logically sound, given the information perceived (independent of whether that information was correct)? (4) Completeness: were any steps skipped in the given answer or left unstated?\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. In your final answer, respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type in {'standard-flex', 'mmvu'}:
        question_type = data['question_type']
        if question_type == 'multiple-choice':
            options_str = ""
            for i, option in enumerate(data['options']):
                options_str += f"{chr(ord('A')+i)}: {option}\n"
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:" 
        else:
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nAnswer the following question based on the video and the subtitles. The answer is short. Please directly respond with the short answer. \n\nQuestion.\n{data['question']}\n\nThe answer is:" 
    elif prompt_type == 'qa-only-flex':
        question_type = data['question_type']
        if question_type == 'multiple-choice':
            options_str = ""
            for i, option in enumerate(data['options']):
                options_str += f"{chr(ord('A')+i)}: {option}\n"
            prompt = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:" 
        else:
            prompt = f"Answer the following question.The answer is short. Please directly respond with the short answer.\n\nQuestion.\n{data['question']}\n\nThe answer is:"
    elif prompt_type == 'subtitle-only-flex':
        question_type = data['question_type']
        if question_type == 'multiple-choice':
            options_str = ""
            for i, option in enumerate(data['options']):
                options_str += f"{chr(ord('A')+i)}: {option}\n"
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
        else:
            prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nAnswer the following question based on the video and the subtitles. The answer is short. Please directly respond with the short answer. \n\nQuestion.\n{data['question']}\n\nThe answer is:" 
    elif prompt_type in {'standard-unified'}:
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nAnswer the question based on all the provided information. If the options are given, please select the most accurate answer. In this case, please respond with only the letter (A, B, C, D, E, etc.) of the correct option. However, if the options are not given, please directly answer the question. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'subtitle-only':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'subtitle-only-unified':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nAnswer the question based on all the provided information. If the options are given, please select the most accurate answer. In this case, please respond with only the letter (A, B, C, D, E, etc.) of the correct option. However, if the options are not given, please directly answer the question. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'qa-only':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'qa-only-unified':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"Answer the question based on all the provided information. If the options are given, please select the most accurate answer. In this case, please respond with only the letter (A, B, C, D, E, etc.) of the correct option. However, if the options are not given, please directly answer the question. In your final response, please only include only the answer without explanation.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'caption-only':
        options_str = ""
        for i, option in enumerate(data['options']):
            options_str += f"{chr(ord('A')+i)}: {option}\n"
        prompt = f"The video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, etc.) of the correct option.\n\nQuestion.\n{data['question']}\n\nOptions.\n{options_str}\n\nThe answer is:"
    elif prompt_type == 'cgbench-miou':
        prompt = f"This video's subtitles are listed below:\n{data['subtitle']}\n\nThe video's captions are listed below. Each caption describes a {clip_length} seconds clip.\n{data['caption']}\n\nYour task is to determine in which intervals the 'clue intervals' exist that contain visual information needed to answer the question.\nHere is the question.\n{data['question']}\n\nOnly output the answer in the following format:\n[[start1, end1], [start2, end2], ...]\nIn this output format, each 'start' and 'end' represents the beginning and end of an interval in seconds (integer) where relevant clues can be found.\nYou must provide at least one interval and at most five intervals.\nHere is one example output.\n[[5, 7]]\nHere is another example output.\n[[200, 207], [209, 213], [214, 220]]"
    else:
        raise NotImplementedError(f"Prompt type {prompt_type} not implemented")
    return [prompt]