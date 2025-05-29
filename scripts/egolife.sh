# Egolife subtitle+caption8
python main.py \
--dataset egolife \
--output_base_path output/egolife/sub+cap8_r1 \
--caption_path data/egolife/captions_8s \
--clip_length 8 \
--stride 1 \
--num_workers 64 \
--model deepseek-reasoner \
--prompt_type egolife \
--anno_path data/egolife/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
--subtitle_path data/egolife/EgoLife/EgoLifeCap/Transcript/A1_JAKE \
--api_key $API_KEY