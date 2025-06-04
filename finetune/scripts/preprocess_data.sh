#!/bin/bash

DATA_SETTING=$1
MODE_TYPE=$2
TOKENIZER_MODEL=$3
AUDIO_PROMPT_MODES=($4)
if [ -z "$4" ]; then
    AUDIO_PROMPT_MODES=('dual' 'inst' 'vocal' 'mixture')
fi

if [ -z "$DATA_SETTING" ] || [ -z "$MODE_TYPE" ]; then
    echo "Usage: $0 <setting> <mode_type>"
    echo "  <setting>: e.g., dummy"
    echo "  <mode_type>: cot or icl_cot"
    exit 1
fi

# Common settings based on DATA_SETTING
if [ "$DATA_SETTING" == "dummy" ]; then
       DATA_ROOT=example
       NAME_PREFIX=dummy.msa.xcodec_16k
       CODEC_TYPE=xcodec
       INSTRUCTION="Generate music from the given lyrics segment by segment."
       ORDER=textfirst
       DROPOUT=0.0
       KEEP_SEQUENTIAL_SAMPLES=true
       QUANTIZER_BEGIN_IDX=0
       NUM_QUANTIZERS=1
else
    echo "Invalid setting: $DATA_SETTING"
    exit 1
fi

JSONL_NAME=jsonl/$NAME_PREFIX.jsonl

# Mode-specific settings and execution
if [ "$MODE_TYPE" == "cot" ]; then
    echo "Running in 'cot' mode..."
    NAME_SUFFIX=stage_1_token_level_interleave_cot_xcodec
    MMAP_NAME=mmap/${NAME_PREFIX}_${NAME_SUFFIX}_$ORDER

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl # Use -f to avoid error if files don't exist
    mkdir -p $DATA_ROOT/$MMAP_NAME

    args="python core/preprocess_data_conditional_xcodec_segment.py \
           --input $DATA_ROOT/$JSONL_NAME \
           --output-prefix $DATA_ROOT/$MMAP_NAME \
           --tokenizer-model $TOKENIZER_MODEL \
           --tokenizer-type MMSentencePieceTokenizer \
           --codec-type $CODEC_TYPE \
           --workers 8 \
           --partitions 1 \
           --instruction \"$INSTRUCTION\" \
           --instruction-dropout-rate $DROPOUT \
           --order $ORDER \
           --append-eod \
           --quantizer-begin $QUANTIZER_BEGIN_IDX \
           --n-quantizer $NUM_QUANTIZERS \
           --use-token-level-interleave \
           --keep-sequential-samples \
           --cot
           "

    echo "$args"
    sleep 5
    eval $args

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl # Use -f
    rm -f $DATA_ROOT/${MMAP_NAME}_*_text_document.bin # Use -f
    rm -f $DATA_ROOT/${MMAP_NAME}_*_text_document.idx # Use -f

elif [ "$MODE_TYPE" == "icl_cot" ]; then
    echo "Running in 'icl_cot' mode..."
    NAME_SUFFIX=stage_1_token_level_interleave_long_prompt_msa
    MMAP_NAME=mmap/${NAME_PREFIX}_${NAME_SUFFIX}_$ORDER # Define MMAP_NAME base for this mode
    PROMPT_LEN=30

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl # Use -f
    mkdir -p $DATA_ROOT/$MMAP_NAME # Ensure base MMAP dir exists

    
    for mode in "${AUDIO_PROMPT_MODES[@]}"; do
           echo "Processing mode: $mode"
           MODE_MMAP_NAME=${MMAP_NAME}_${mode} # Mode specific path
           mkdir -p $DATA_ROOT/$MODE_MMAP_NAME # Ensure mode-specific dir exists

           args="python core/preprocess_data_conditional_xcodec_segment.py \
                  --input $DATA_ROOT/$JSONL_NAME \
                  --output-prefix $DATA_ROOT/$MODE_MMAP_NAME \
                  --tokenizer-model $TOKENIZER_MODEL \
                  --tokenizer-type MMSentencePieceTokenizer \
                  --codec-type $CODEC_TYPE \
                  --workers 8 \
                  --partitions 1 \
                  --instruction \"$INSTRUCTION\" \
                  --instruction-dropout-rate $DROPOUT \
                  --order $ORDER \
                  --append-eod \
                  --quantizer-begin $QUANTIZER_BEGIN_IDX \
                  --n-quantizer $NUM_QUANTIZERS \
                  --cot \
                  --use-token-level-interleave \
                  --use-audio-icl \
                  --audio-prompt-mode $mode \
                  --audio-prompt-len $PROMPT_LEN \
                  --keep-sequential-samples
                  "

           echo "$args"
           sleep 5
           eval $args

           # Clean up mode-specific files
           rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl # Use -f
           rm -f $DATA_ROOT/${MODE_MMAP_NAME}_*_text_document.bin # Use -f
           rm -f $DATA_ROOT/${MODE_MMAP_NAME}_*_text_document.idx # Use -f
    done

else
    echo "Invalid mode_type: $MODE_TYPE. Use 'cot' or 'icl_cot'."
    exit 1
fi

echo "Preprocessing finished for setting '$DATA_SETTING' and mode_type '$MODE_TYPE'." 