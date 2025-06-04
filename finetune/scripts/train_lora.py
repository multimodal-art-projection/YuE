# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import time
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
import wandb
from peft import LoraConfig, get_peft_model
from core.arguments import parse_args
from core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_GLOBAL_TOKENIZER = None

def is_dataset_built_on_rank():
    # return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0
    return True

def core_gpt_dataset_config_from_args(args):
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        eod_id=_GLOBAL_TOKENIZER.vocab['<EOD>'],
        enable_shuffle=args.enable_shuffle,
    )

def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                            args.model_name_or_path, 
                            model_max_length=args.model_max_length, 
                            padding_side="right")
    return _GLOBAL_TOKENIZER

def build_train_valid_test_datasets(args):
    """Build the train, validation, and test datasets."""
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                 eval_iters * args.global_batch_size,
                                 test_iters * args.global_batch_size]

    logger.info("> Building train, validation, and test datasets...")
    try:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            core_gpt_dataset_config_from_args(args)
        ).build()
        logger.info("> Finished creating datasets")
        return train_ds, valid_ds, test_ds
    except Exception as e:
        logger.error(f"Failed to build datasets: {e}")
        raise

def _compile_dependencies():
    """Compile dataset C++ code."""
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        logger.info("> Compiling dataset index builder...")
        try:
            from core.datasets.utils import compile_helpers
            compile_helpers()
            logger.info(
                f">>> Done with dataset index builder. Compilation time: {time.time() - start_time:.3f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to compile helpers: {e}")
            raise

def setup_distributed_training():
    """Setup distributed training environment."""
    try:
        # Initialize process group for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        if world_size > 1:
            # Multi-GPU setup
            torch.cuda.set_device(local_rank)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            logger.info(f"Distributed training initialized with world size: {world_size}, local rank: {local_rank}")
        else:
            # Single GPU setup
            logger.info(f"Running on a single GPU (device {local_rank})")
            torch.cuda.set_device(local_rank)
        
        return local_rank
    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        raise

def create_and_configure_model(args):
    """Create and configure the model with LoRA."""
    try:
        if args.fp16:
            assert not args.bf16
            args.params_dtype = torch.half
        if args.bf16:
            assert not args.fp16
            args.params_dtype = torch.bfloat16
        logger.info(f"Loading base model from {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.params_dtype,
            cache_dir=args.cache_dir
        )
        
        logger.info(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to create and configure model: {e}")
        raise

def main():
    # Setup distributed training
    local_rank = setup_distributed_training()
    
    # Compile dependencies after initializing distributed group
    _compile_dependencies()
    
    # Parse arguments
    args = parse_args()
    
    # Build tokenizer
    _build_tokenizer(args)
    
    # Build datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(args)
    
    # Create and configure model
    model = create_and_configure_model(args)
    
    # Setup training arguments
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(args.__dict__, allow_extra_keys=True)[0]
    
    # Initialize wandb if specified
    is_main_process = torch.distributed.get_rank() == 0
    if args.report_to == "wandb" and is_main_process:
        try:
            wandb.init(
                project=args.wandb_project or "YuE-finetune",
                config=vars(args),
                name=args.run_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=_GLOBAL_TOKENIZER,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=default_data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    if is_main_process:
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        _GLOBAL_TOKENIZER.save_pretrained(training_args.output_dir)
        logger.info("Training completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise