# NVILA 8s captions + DeepSeek-R1
python main.py \
--dataset videomme \
--output_base_path output/videomme/sub+cap8_r1 \
--caption_path data/videomme/captions_8s \
--clip_length 8 \
--stride 1 \
--model deepseek-reasoner \
--num_workers 64 \
--prompt_type videomme \
--subtitle_path data/videomme/subtitles \
--anno_path hf://datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# NVILA 64s captions + Llama-4-scout (local)
# Running local LLMs requires very large GPU memory. Optionally, you can use API service like Lambda. (see below)
python main.py \
--dataset videomme \
--output_base_path output/videomme/sub+cap8_llama4scout \
--caption_path data/videomme/captions_64s \
--clip_length 64 \
--stride 1 \
--model meta-llama/Llama-4-Scout-17B-16E-Instruct \
--single_process \
--prompt_type videomme \
--subtitle_path data/videomme/subtitles \
--anno_path hf://datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet \
--hf_token $HF_TOKEN

# NVILA 64s captions + Llama-4-scout (Lambda API)
python main.py \
--dataset videomme \
--output_base_path output/videomme/sub+cap8_llama4scout_lambda \
--caption_path data/videomme/captions_64s \
--clip_length 64 \
--stride 1 \
--model llama-4-scout-17b-16e-instruct \
--endpoint lambda \
--api_url https://api.lambda.ai/v1/chat/completions \
--num_workers 8 \
--prompt_type videomme \
--subtitle_path data/videomme/subtitles \
--anno_path hf://datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# NVILA 1s captions + DeepSeek-R1
# Note that there are many videos which exceed LLM's context length (64K).
# For those videos, one option is to use the 8s captions instead instead. See --backup_path.
python main.py \
--dataset videomme \
--output_base_path output/videomme/sub+cap1_r1 \
--caption_path data/videomme/captions_1s \
--clip_length 1 \
--stride 1 \
--model deepseek-reasoner \
--num_workers 32 \
--prompt_type videomme \
--subtitle_path data/videomme/subtitles \
--anno_path hf://datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet \
--backup_path output/videomme/sub+cap8_r1 \
--api_key $API_KEY \
--hf_token $HF_TOKEN