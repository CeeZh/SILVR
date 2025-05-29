# CinePile Test subtitle+caption clip16
python main.py \
--dataset cinepile \
--output_base_path output/cinepile/sub+cap16_r1 \
--caption_path data/cinepile/captions_16s \
--clip_length 16 \
--stride 1 \
--num_workers 128 \
--prompt_type cinepile \
--anno_path hf://datasets/tomg-group-umd/cinepile/v2/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN


# CinePile Test subtitle+caption clip1
python main.py \
--dataset cinepile \
--output_base_path output/cinepile/sub+cap1_r1 \
--caption_path data/cinepile/captions_1s \
--clip_length 1 \
--stride 1 \
--num_workers 64 \
--prompt_type cinepile \
--anno_path hf://datasets/tomg-group-umd/cinepile/v2/test-00000-of-00001.parquet \
--api_key $API_KEY \
--hf_token $HF_TOKEN
