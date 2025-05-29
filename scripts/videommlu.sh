# VideoMMLU subtitle+caption8
# Note that this script will run Qwen-2.5-72B-Instruct to evaluate the results.
# You can set --disable_eval to skip evaluation.
python main.py \
--dataset videommlu \
--output_base_path output/videommlu/sub+cap8_r1 \
--caption_path data/videommlu/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--model deepseek-reasoner \
--prompt_type videommlu \
--anno_path data/videommlu/Video-MMLU/video_mmlu.jsonl \
--subtitle_path data/videommlu/Video-MMLU/transcript/transcripts_punctuated.json \
--videommlu_category_file data/videommlu/Video-MMLU/video_sources.jsonl \
--api_key $API_KEY