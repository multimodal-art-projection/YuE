export CUDA_VISIBLE_DEVICES=4

export VLLM_NUM_PROCS=1
export VLLM_USE_V1=0

python inference_vllm_pipeline_batching.py \
    --cuda_idx 0 \
    --stage1_model /map-vepfs/zhengxuan/YuE-s1-7B-anneal-en-cot \
    --stage2_model /map-vepfs/zhengxuan/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 6 \
    --stage2_batch_size 16 \
    --output_dir ../output/vllm_cfg1.5_1.2_seed233 \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --seed 233 \
    --request_batch_size 2 \
    --prompts_config_path /map-vepfs/zhengxuan/yue/prompts_config