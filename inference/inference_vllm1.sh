export CUDA_VISIBLE_DEVICES=0

export VLLM_NUM_PROCS=1
export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 python infer_stage1_vllm.py \
    --output_length 30 \
    --dtype bf16