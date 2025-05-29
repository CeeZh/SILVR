# MMWorld subtitle+caption clip8
python main.py \
--dataset mmworld \
--output_base_path output/mmworld/sub+cap8_r1 \
--caption_path data/mmworld/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--prompt_type mmworld \
--anno_path hf://datasets/MMWorld/MMWorld/data/train-00000-of-00001.parquet \
--subtitle_path data/mmworld/subtitles \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# MMWorld subtitle+caption clip1
python main.py \
--dataset mmworld \
--output_base_path output/mmworld/sub+cap1_r1 \
--caption_path data/mmworld/captions_1s \
--clip_length 1 \
--stride 1 \
--num_workers 64 \
--prompt_type mmworld \
--anno_path hf://datasets/MMWorld/MMWorld/data/train-00000-of-00001.parquet \
--subtitle_path data/mmworld/subtitles \
--backup_path output/mmworld/sub+cap8_r1/logs \
--api_key $API_KEY \
--hf_token $HF_TOKEN
