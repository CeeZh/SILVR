# VideoMMMU clip 8 Adaptation
python main.py \
--dataset videommmu \
--output_base_path output/videommmu/sub+cap8_r1/Adaptation \
--caption_path data/videommmu/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--prompt_type videommmu \
--subtitle_path data/videommmu/subtitles \
--anno_path hf://datasets/lmms-lab/VideoMMMU/Adaptation/test-00000-of-00001.parquet \
--image_caption_path data/videommmu/image_captions \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# VideoMMMU clip 8 Comprehension
python main.py \
--dataset videommmu \
--output_base_path output/videommmu/sub+cap8_r1/Comprehension \
--caption_path data/videommmu/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--prompt_type videommmu \
--subtitle_path data/videommmu/subtitles \
--anno_path hf://datasets/lmms-lab/VideoMMMU/Comprehension/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# VideoMMMU clip 8 Perception
python main.py \
--dataset videommmu \
--output_base_path output/videommmu/sub+cap8_r1/Perception \
--caption_path data/videommmu/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--prompt_type videommmu \
--subtitle_path data/videommmu/subtitles \
--anno_path hf://datasets/lmms-lab/VideoMMMU/Perception/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN
