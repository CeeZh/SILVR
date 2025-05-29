# HourVideo dev subtitle+caption clip60 
# This script will generate the submission file for the dev phase. Then you need to upload the submission file to the evalai website for evaluation.
python main.py \
--dataset hourvideo \
--output_base_path output/videommlu/sub+cap60_r1/dev \
--caption_path data/hourvideo/HourVideo/socratic_models_captions/GPT-4-TURBO \
--clip_length 60 \
--stride 1 \
--num_workers 128 \
--prompt_type hourvideo \
--anno_path hf://datasets/HourVideo/HourVideo/v1.0_release/parquet/dev_v1.0.parquet \
--subtitle_path data/hourvideo/subtitles \
--submission_reference_path data/hourvideo/HourVideo/evalai_submission_examples/dev_phase_with_dummy_answers.json \
--hourvideo_image_caption_path data/hourvideo/question_captions \
--api_key $API_KEY \
--hf_token $HF_TOKEN