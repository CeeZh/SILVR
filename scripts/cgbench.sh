# CGBench long-acc. subtitle+caption32
python main.py \
--dataset cgbench \
--output_base_path output/cgbench/sub+cap32_r1 \
--caption_path data/cgbench/captions_32s \
--clip_length 32 \
--stride 1 \
--num_workers 64 \
--model deepseek-reasoner \
--prompt_type cgbench \
--anno_path hf://datasets/CG-Bench/CG-Bench/cgbench.json \
--subtitle_path data/cgbench/subtitles \
--api_key $API_KEY \
--hf_token $HF_TOKEN



# CGBench-miou subtitle+caption32
python main.py \
--dataset cgbench \
--cgbench_task miou \
--output_base_path output/cgbench-miou/sub+cap32_r1 \
--caption_path data/cgbench/captions_32s \
--clip_length 32 \
--stride 1 \
--num_workers 64 \
--model deepseek-reasoner \
--prompt_type cgbench-miou \
--anno_path hf://datasets/CG-Bench/CG-Bench/cgbench.json \
--subtitle_path data/cgbench/subtitles \
--api_key $API_KEY \
--hf_token $HF_TOKEN
