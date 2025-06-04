#!/bin/bash

# ==============================
# YuE Fine-tuning Script
# ==============================

# Help information
print_help() {
  echo "========================================================"
  echo "YuE Fine-tuning Script Help"
  echo "========================================================"
  echo "Before running this script, please update the following variables:"
  echo ""
  echo "1. Data paths:"
  echo "   DATA_PATH - Replace <weight_and_path_to_data_X> with actual weights and data paths"
  echo "   DATA_CACHE_PATH - Replace <path_to_data_cache> with actual cache directory"
  echo ""
  echo "2. Model configuration:"
  echo "   TOKENIZER_MODEL_PATH - Replace <path_to_tokenizer_model> with actual tokenizer path"
  echo "   MODEL_CACHE_DIR - Replace <path_to_model_cache> with actual cache directory"
  echo "   OUTPUT_DIR - Replace <path_to_output_dir> with actual output directory"
  echo ""
  echo "3. If using WandB:"
  echo "   WANDB_API_KEY - Replace <your_wandb_api_key> with your actual API key"
  echo ""
  echo "Example usage:"
  echo "  DATA_PATH=\"data1-weight /path/to/data1 data2-weight /path/to/data2\""
  echo "  DATA_CACHE_PATH=\"/path/to/cache\""
  echo "  TOKENIZER_MODEL_PATH=\"/path/to/tokenizer\""
  echo "  MODEL_CACHE_DIR=\"/path/to/model/cache\""
  echo "  OUTPUT_DIR=\"/path/to/output\""
  echo "  WANDB_API_KEY=\"your-actual-wandb-key\""
  echo "========================================================"
  exit 1
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  print_help
fi

# Check for placeholder values
check_placeholders() {
  local has_placeholders=false
  
  if [[ "$DATA_PATH" == *"<weight_and_path_to_data"* ]]; then
    echo "Error: Please set actual weight and data paths in DATA_PATH variable."
    has_placeholders=true
  fi
  
  if [[ "$DATA_CACHE_PATH" == *"<path_to_data_cache>"* ]]; then
    echo "Error: Please set actual data cache path in DATA_CACHE_PATH variable."
    has_placeholders=true
  fi
  
  if [[ "$TOKENIZER_MODEL_PATH" == *"<path_to_tokenizer_model>"* ]]; then
    echo "Error: Please set actual tokenizer model path in TOKENIZER_MODEL_PATH variable."
    has_placeholders=true
  fi
  
  if [[ "$MODEL_CACHE_DIR" == *"<path_to_model_cache>"* ]]; then
    echo "Error: Please set actual model cache directory in MODEL_CACHE_DIR variable."
    has_placeholders=true
  fi
  
  if [[ "$OUTPUT_DIR" == *"<path_to_output_dir>"* ]]; then
    echo "Error: Please set actual output directory in OUTPUT_DIR variable."
    has_placeholders=true
  fi
  
  if [[ "$USE_WANDB" == "true" && "$WANDB_API_KEY" == *"<your_wandb_api_key>"* ]]; then
    echo "Error: Please set actual WandB API key in WANDB_API_KEY variable or disable WandB."
    has_placeholders=true
  fi
  
  if [[ "$has_placeholders" == "true" ]]; then
    echo ""
    echo "Please update the script with your actual paths and values."
    echo "Run './scripts/run_finetune.sh --help' for more information."
    exit 1
  fi
}

# Exit on error
set -e

# Check if we're in the finetune directory
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "finetune" ]; then
  echo "Error: This script must be run from the finetune/ directory"
  echo "Current directory: $PWD"
  echo "Please change to the finetune directory and try again"
  exit 1
fi

# ==============================
# Configuration Parameters
# ==============================

# Hardware configuration
NUM_GPUS=8
MASTER_PORT=9999
# Uncomment and modify if you need specific GPUs
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# Training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$((NUM_GPUS*PER_DEVICE_TRAIN_BATCH_SIZE))
USE_BF16=true
SEQ_LENGTH=8192
TRAIN_ITERS=150
NUM_TRAIN_EPOCHS=10

# Data paths (replace with your actual paths)
DATA_PATH="<weight_and_path_to_data_X>"     
DATA_CACHE_PATH="<path_to_tokenizer_model>"

# Set comma-separated list of proportions for training, validation, and test split
DATA_SPLIT="900,50,50"

# Model configuration
TOKENIZER_MODEL_PATH="<path_to_tokenizer_model>"
MODEL_NAME="m-a-p/YuE-s1-7B-anneal-en-cot"
MODEL_CACHE_DIR="<path_to_model_cache>"
OUTPUT_DIR="<path_to_output_dir>"
DEEPSPEED_CONFIG=config/ds_config_zero2.json

# LoRA configuration
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"
# Logging configuration
LOGGING_STEPS=5
SAVE_STEPS=5
USE_WANDB=true
WANDB_API_KEY="<your_wandb_api_key>"
RUN_NAME="YuE-ft-lora"

# ==============================
# Environment Setup
# ==============================

# Check for placeholder values
check_placeholders

# Export environment variables
export WANDB_API_KEY=$WANDB_API_KEY
export PYTHONPATH=$PWD:$PYTHONPATH

# Print configuration
echo "==============================================="
echo "YuE Fine-tuning Configuration:"
echo "==============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Training epochs: $NUM_TRAIN_EPOCHS"
echo "==============================================="

# ==============================
# Build and Execute Command
# ==============================

# Base command
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT scripts/train_lora.py \
    --seq-length $SEQ_LENGTH \
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --split $DATA_SPLIT \
    --tokenizer-model $TOKENIZER_MODEL_PATH \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --per-device-train-batch-size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per-device-eval-batch-size $PER_DEVICE_EVAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --num-train-epochs $NUM_TRAIN_EPOCHS \
    --logging-steps $LOGGING_STEPS \
    --save-steps $SAVE_STEPS \
    --deepspeed $DEEPSPEED_CONFIG"

# Add conditional arguments
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --report-to wandb --run-name \"$RUN_NAME\""
elif [ "$USE_WANDB" = false ]; then
    CMD="$CMD --report-to none"
fi

CMD="$CMD \
    --model-name-or-path \"$MODEL_NAME\" \
    --cache-dir $MODEL_CACHE_DIR \
    --output-dir $OUTPUT_DIR \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --lora-target-modules $LORA_TARGET_MODULES"

if [ "$USE_BF16" = true ]; then
    CMD="$CMD --bf16"
fi

# Execute the command
echo "Running command: $CMD"
echo "==============================================="
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "==============================================="
    echo "Fine-tuning completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    echo "==============================================="
else
    echo "==============================================="
    echo "Error: Fine-tuning failed with exit code $?"
    echo "==============================================="
    exit 1
fi