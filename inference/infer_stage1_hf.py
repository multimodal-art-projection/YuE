"""
Dependencies:
    transformers 4.42.0
    torch        2.1.2

Example usage:
    python infer_stage1_hf.py \
        --model_path "m-a-p/YuE-s1-7B-anneal-en-cot" \
        --cache_dir $YOUR_CACHE_DIR \
        --output_dir $YOUR_OUTPUT_DIR \
        --seed 42 \
        --dtype "bf16" \
        --output_length 300 
"""

import os
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.file_download import try_to_load_from_cache

# Constants for sampling parameters
default_sampling_params = {
    "top_p": 0.93,
    "temperature": 1.0,
    "repetition_penalty": 1.2,
}

def seed_everything(seed=42): 
    """Fixes random seed for reproducibility."""
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HFInference:
    def __init__(self, model_path: str, cache_dir: str, dtype: torch.dtype, output_length: int):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.output_length = output_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._init_model()
        
    def _init_model(self):
        print("Initializing Hugging Face model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, padding_side="left", cache_dir=self.cache_dir
        )
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device).eval()
    
    def generate(self, prompt_text: str, **overrides) -> Tuple[torch.Tensor, str]:
        """Generates output using Hugging Face model."""
        print("Generating with Hugging Face model.....")
        params = {**default_sampling_params, **overrides}
        do_sample = params["temperature"] > 0
        
        # 添加特殊token
        prompt_ids = self.tokenizer.encode(prompt_text) + [32001, 32016]
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.hf_model.generate(
                input_ids=input_ids,
                min_new_tokens=self.output_length,
                max_new_tokens=self.output_length,
                top_p=params["top_p"],
                top_k=50,
                temperature=params["temperature"],
                repetition_penalty=params["repetition_penalty"],
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                guidance_scale=1.0,
            )
        
        # 获取生成的token
        generated_ids = outputs.sequences[0][input_ids.shape[-1]:]
        import pdb; pdb.set_trace()
        
        return generated_ids

def main():
    parser = argparse.ArgumentParser(description="Hugging Face model inference.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--model_path', type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help='Path to the model')
    parser.add_argument('--cache_dir', type=str, default="/home/jianweiyu/.cache/huggingface/hub/", help='Cache directory')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32'], help='Model data type')
    parser.add_argument('--output_length', type=int, default=300, help='Output sequence length')
    parser.add_argument('--output_dir', type=str, default="output/cfg", help='Output directory')
    args = parser.parse_args()
    
    seed_everything(args.seed)

    if args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    
    model = HFInference(args.model_path, args.cache_dir, dtype, args.output_length)
    prompt_text = """Generate music from the given lyrics segment by segment.
[Genre] male crisp vocal R&B Love Romance
In the heart of knowledge, where dreams take flight,
A vision unfolds, bathed in radiant light."""

    generated_ids, generated_text = model.generate(prompt_text)
    
    # 保存生成的token ids
    np.save(os.path.join(args.output_dir, "hf_output.npy"), generated_ids.cpu().numpy())
    
    # 保存生成的文本
    with open(os.path.join(args.output_dir, "hf_output.txt"), "w") as f:
        f.write(generated_text)
    
    print(f"Generated text saved to {os.path.join(args.output_dir, 'hf_output.txt')}")
    print(f"Token IDs saved to {os.path.join(args.output_dir, 'hf_output.npy')}")

if __name__ == "__main__":
    main()
