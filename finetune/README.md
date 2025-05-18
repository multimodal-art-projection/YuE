# YuE Finetuning Guide

This guide walks you through the process of finetuning the YuE model using your own data.

## Table of Contents
1. [Data Preparation](#step-1-data-preparation)
2. [Training Data Configuration](#step-2-training-data-configuration)
3. [Model Finetuning](#step-3-model-finetuning)

## Requirements

- Python 3.10 is recommended
- PyTorch 2.4 is recommended
- CUDA 12.1+ is recommended

```bash
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/finetune/
conda create -n yue-ft python=3.10
conda activate yue-ft
pip install -r requirements.txt
```

## Step 1: Data Preparation

### Required Data Structure

Your data should be organized in the following structure:
```
example/
├── jsonl/     # Source JSONL files
├── mmap/      # Generated Megatron binary files
└── npy/       # Discrete audio codes (numpy arrays) from xcodec
```

### JSONL File Format

Each JSONL file should contain entries in the following format:

```json
{
    "id": "1",
    "codec": "example/npy/dummy.npy",                    // Raw audio codes
    "vocals_codec": "example/npy/dummy.Vocals.npy",      // Vocal track codes
    "instrumental_codec": "example/npy/dummy.Instrumental.npy",  // Instrumental track codes
    "audio_length_in_sec": 85.16,                        // Audio duration in seconds
    "msa": [                                             // Music Structure Analysis
        {
            "start": 0,
            "end": 13.93,
            "label": "intro"
        }
    ],
    "genres": "male, youth, powerful, charismatic, rock, punk",  // Tags for gender, age, genre, mood, timbre
    "splitted_lyrics": {
        "segmented_lyrics": [
            {
                "offset": 0,
                "duration": 13.93,
                "codec_frame_start": 0,
                "codec_frame_end": 696,
                "line_content": "[intro]\n\n"
            }
        ]
    }
}
```

### Converting to Megatron Binary Format

1. Navigate to the finetune directory:
```bash
cd finetune/
```

2. Run the preprocessing script:
```bash
# For Chain-of-Thought (CoT) dataset
bash scripts/preprocess_data.sh dummy cot $TOKENIZER_MODEL

# For In-Context Learning (ICL) dataset
bash scripts/preprocess_data.sh dummy icl_cot $TOKENIZER_MODEL
```

> **Note**: For music structure analysis and track separation, refer to [openl2s](https://github.com/a43992899/openl2s).

## Step 2: Training Data Configuration

### Counting Dataset Tokens

1. Navigate to the finetune directory:
```bash
cd finetune/
```

2. Run the token counting script:
```bash
bash scripts/count_tokens.sh ./example/mmap/
```

The results will be saved in `finetune/count_token_logs/`. This process may take several minutes for large datasets.

### Configuring Data Mixture

1. Create a configuration file (e.g., `finetune/example/dummy_data_mixture_cfg.yml`) with the following parameters:
   - `TOKEN_COUNT_LOG_DIR`: Directory containing token count logs
   - `GLOBAL_BATCH_SIZE`: Total batch size for training
   - `SEQ_LEN`: Maximum context window size
   - `{NUM}_ROUND`: Number of times to repeat each dataset

2. Generate training parameters:
```bash
cd finetune/
python core/parse_mixture.py -c example/dummy_data_mixture_cfg.yml
```

The script will output:
- `DATA_PATH`: Paths to your training data (copy to training script)
- `TRAIN_ITERS`: Number of training iterations
- Total token count

## Step 3: Model Finetuning

YuE supports finetuning using LoRA (Low-Rank Adaptation), which significantly reduces memory requirements while maintaining performance.

### Configuring the Finetuning Script

1. Edit the `scripts/run_finetune.sh` script to configure your finetuning run:

```bash
# Update data paths
# Accepted formats for DATA_PATH:
#   1) a single path: "/path/to/data"
#   2) multiple datasets with weights: "100 /path/to/data1 200 /path/to/data2 ..."
# You can copy DATA_PATH from the output of core/parse_mixture.py in Step 2
DATA_PATH="data1-weight /path/to/data1 data2-weight /path/to/data2"
DATA_CACHE_PATH="/path/to/your/cache"

# Set comma-separated list of proportions for train/val/test split
DATA_SPLIT="900,50,50"

# Set model paths
TOKENIZER_MODEL_PATH="/path/to/tokenizer"
MODEL_NAME="m-a-p/YuE-s1-7B-anneal-en-cot"  # or your local model path
MODEL_CACHE_DIR="/path/to/model/cache"
OUTPUT_DIR="/path/to/save/finetuned/model"

# Configure LoRA parameters (optional)
LORA_R=64              # Rank of the LoRA update matrices
LORA_ALPHA=32          # Scaling factor for the LoRA update
LORA_DROPOUT=0.1       # Dropout probability for LoRA layers
```

2. Adjust training hyperparameters as needed:
```bash
# Training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=10
```

### Running the Finetuning Process

```bash
cd finetune/
bash scripts/run_finetune.sh
```

For help with configuring the script:
```bash
bash scripts/run_finetune.sh --help
```

### Monitoring Training

If you've enabled WandB logging (via `USE_WANDB=true`), you can monitor your training progress in real-time through the WandB dashboard.

### Using the Finetuned Model

After training completes, your model will be saved to the specified `OUTPUT_DIR`. You can use this model for inference or further finetuning.