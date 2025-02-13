"""
Dependencies:
    transformers 4.42.0
    vllm         0.4.0
    torch        2.1.2

Example usage:
    python compare_hf_and_vllm.py \
        --model_path "m-a-p/YuE-s1-7B-anneal-en-cot" \
        --cache_dir $YOUR_CACHE_DIR \
        --output_dir $YOUR_OUTPUT_DIR \
        --seed 42 \
        --dtype "bf16" \
        --output_length 300 
"""

import os
import uuid
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from einops import rearrange
from huggingface_hub.file_download import try_to_load_from_cache
from codecmanipulator import CodecManipulator

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

class ModelComparison:
    def __init__(self, model_path: str, cache_dir: str, dtype: torch.dtype, output_length: int):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.output_length = output_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._init_models()
        
    def _init_models(self):
        print("Initializing models...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, padding_side="left", cache_dir=self.cache_dir
        )
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device).eval()
        
        hf_model_cache_dir = self._get_hf_model_cache_dir()
        self.vllm_model = LLM(
            model=hf_model_cache_dir, gpu_memory_utilization=0.8, dtype=self.dtype
        )
    
    def _get_hf_model_cache_dir(self, revision="dc734b661b51e4d51cfa176ad64b72f66d67c291") -> str:
        """Returns the cache directory path for the Hugging Face model."""
        # TODO: get commit hash instead of revision hardcode
        object_id = self.model_path.replace("/", "--")
        repo_cache = os.path.join(self.cache_dir, f"models--{object_id}")
        return os.path.join(repo_cache, "snapshots", revision) if os.path.isdir(repo_cache) else None
    
    def _get_sampling_params(self, **overrides) -> SamplingParams:
        """Creates sampling parameters with possible overrides."""
        params = {**default_sampling_params, **overrides}
        return SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            min_tokens=self.output_length,
            max_tokens=self.output_length,
            logprobs=1,
        )
    
    def generate_hf(self, input_ids: torch.Tensor, **overrides) -> torch.Tensor:
        """Generates output using Hugging Face model."""
        params = {**default_sampling_params, **overrides}
        do_sample = params["temperature"] > 0
        with torch.no_grad():
            ids = self.hf_model.generate(
                input_ids=input_ids,
                min_new_tokens=self.output_length,
                max_new_tokens=self.output_length,
                top_p=params["top_p"],
                temperature=params["temperature"],
                repetition_penalty=params["repetition_penalty"],
                do_sample=do_sample,
            )
        return ids[0][input_ids.shape[-1]:]
    
    def generate_vllm(self, input_ids: List[List[int]]) -> List[int]:
        """Generates output using vLLM."""
        sampling_params = self._get_sampling_params()
        outputs = self.vllm_model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
        return [list(log_prob.keys())[0] for log_prob in outputs[0].outputs[0].logprobs]
    
    def analyze_differences(self, hf_ids: torch.Tensor, vllm_ids: List[int]) -> List[Dict]:
        """Analyzes differences between Hugging Face and vLLM outputs."""
        hf_ids = hf_ids.tolist()
        first_diff_idx = next((i for i in range(min(len(hf_ids), len(vllm_ids))) if hf_ids[i] != vllm_ids[i]), None)
        differences = [
            {
                'position': i,
                'hf_token': self.tokenizer.decode([hf_ids[i]]),
                'vllm_token': self.tokenizer.decode([vllm_ids[i]]),
                'hf_id': hf_ids[i],
                'vllm_id': vllm_ids[i],
            }
            for i in range(min(len(hf_ids), len(vllm_ids))) if hf_ids[i] != vllm_ids[i]
        ]
        print(f"Total differences: {len(differences)} | First difference at position: {first_diff_idx}")
        return differences
    
    def compare_outputs(self, prompt_text: str) -> Tuple[torch.Tensor, List[int], List[Dict]]:
        """Compares outputs from Hugging Face and vLLM models."""
        prompt_ids = self.tokenizer.encode(prompt_text) + [32001, 32016]
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(self.device)
        
        hf_ids = self.generate_hf(input_ids)
        vllm_ids = self.generate_vllm(input_ids.tolist())
        differences = self.analyze_differences(hf_ids, vllm_ids)
        return hf_ids, vllm_ids, differences

def main():
    parser = argparse.ArgumentParser(description="Model comparison between Hugging Face and VLLM.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--model_path', type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help='Path to the model')
    parser.add_argument('--cache_dir', type=str, default="/aifs4su/mmcode/codeclm/opensuno_publish", help='Cache directory')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help='Model data type')
    parser.add_argument('--output_length', type=int, default=300, help='Output sequence length')
    parser.add_argument('--output_dir', type=str, default="/aifs4su/mmcode/codeclm/opensuno_publish/YuE/output", help='Output directory')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    os.makedirs(args.output_dir, exist_ok=True)
    
    comparator = ModelComparison(args.model_path, args.cache_dir, dtype, args.output_length)
    prompt_text = """Generate music from the given lyrics segment by segment.
[Genre] male crisp vocal R&B Love Romance
In the heart of knowledge, where dreams take flight,
A vision unfolds, bathed in radiant light."""

    hf_ids, vllm_ids, differences = comparator.compare_outputs(prompt_text)
    
    np.save(os.path.join(args.output_dir, "hf_output.npy"), hf_ids.cpu().numpy())
    np.save(os.path.join(args.output_dir, "vllm_output.npy"), np.array(vllm_ids))
    # print(differences)

if __name__ == "__main__":
    main()
