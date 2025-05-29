# MMVU subtitle+caption clip8
# Note that MMVU uses GPT-4o to do evaluation. You can set --disable_eval to skip evaluation. Additionally, you can set --disable_inference to skip inference.
python main.py \
--dataset mmvu \
--output_base_path output/mmvu/sub+cap8_r1 \
--caption_path data/mmvu/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--prompt_type mmvu \
--anno_path hf://datasets/yale-nlp/MMVU/validation.json \
--subtitle_path data/mmvu/subtitles \
--api_key $API_KEY \
--hf_token $HF_TOKEN \
--openai_api_key $OPENAI_API_KEY


# MMVU subtitle+caption clip1
# Note that MMVU uses GPT-4o to do evaluation. You can set --disable_eval to skip evaluation. Additionally, you can set --disable_inference to skip inference.
python main.py \
--dataset mmvu \
--output_base_path output/mmvu/sub+cap1_r1 \
--caption_path data/mmvu/captions_1s \
--clip_length 1 \
--stride 1 \
--num_workers 64 \
--prompt_type mmvu \
--anno_path hf://datasets/yale-nlp/MMVU/validation.json \
--subtitle_path data/mmvu/subtitles \
--backup_path output/mmvu/sub+cap8_r1/logs \
--api_key $API_KEY \
--hf_token $HF_TOKEN \
--openai_api_key $OPENAI_API_KEY
