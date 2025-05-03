# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import gc
from vllm import LLM, SamplingParams
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import ray
import contextlib
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

#assert Version(ray.__version__) >= Version("2.44.1"), "Ray version must be at least 2.44.1"

#region 参数配置
parser = argparse.ArgumentParser()

# 新增批量处理参数
parser.add_argument("--request_batch_size", type=int, default=8,
                   help="Maximum number of prompts to process in one batch")
parser.add_argument("--prompts_config_path", type=str, required=True,
                   help="Directory containing JSON config files for each prompt request")

# 原有参数配置
# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help="The model checkpoint path or identifier for the Stage 1 model.")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general", help="The model checkpoint path or identifier for the Stage 2 model.")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.")
parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during the generation.")
parser.add_argument("--stage2_batch_size", type=int, default=16, help="The batch size used in Stage 2 inference.")
# Prompt
parser.add_argument("--genre_txt", type=str, required=True, help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.")
parser.add_argument("--lyrics_txt", type=str, required=True, help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.")
parser.add_argument("--use_audio_prompt", action="store_true", help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.")
parser.add_argument("--audio_prompt_path", type=str, default="", help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0, help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--use_dual_tracks_prompt", action="store_true", help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.")
parser.add_argument("--vocal_track_prompt_path", type=str, default="", help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="", help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
# Output 
parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=42, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for xcodec configurations.')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the xcodec checkpoint.')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', help='Path to Vocos config file.')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')


args = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BatchProcessor:
    def __init__(self, args):
        self.args = args
        self.cuda_idx = self.args.cuda_idx
        self.device = torch.device(f"cuda:{self.cuda_idx}" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = self.args.max_new_tokens
        self._init_codec_model()
        
    def _init_codec_model(self):
        self.mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)
        model_config = OmegaConf.load(self.args.basic_model_config)
        self.codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(self.device)
        parameter_dict = torch.load(self.args.resume_path, map_location='cpu', weights_only=False)
        self.codec_model.load_state_dict(parameter_dict['codec_model'])
        self.codec_model.eval()

    def process_batch(self, batch_configs):
        #ray.init(log_to_driver=False)
        try:
            # Stage1处理
            stage1_results = self._process_stage1(batch_configs)
            
            # Stage2处理
            stage2_results = self._process_stage2(stage1_results, batch_configs)
            
            # 音频处理
            self._process_audio_outputs(stage2_results)
            
        finally:
            #ray.shutdown()
            if not self.args.disable_offload_model:           
                torch.cuda.empty_cache()              

    def _process_stage1(self, batch_configs):  
        self.stage1_model = LLM(
            model=self.args.stage1_model, 
            dtype=torch.bfloat16, 
            enable_chunked_prefill=True, 
            max_num_batched_tokens=4096, 
            max_model_len=16384, 
            enforce_eager=True,      # 禁用Ray异步执行
            tensor_parallel_size=1,  # 禁用分布式
            gpu_memory_utilization=0.9,   # 显存利用率调整
        )
        
        outputs = []
        # 特殊token处理
        start_of_segment = self.mmtokenizer.tokenize('[start_of_segment]')
        end_of_segment = self.mmtokenizer.tokenize('[end_of_segment]')
        soa_token = self.mmtokenizer.soa
        eoa_token = self.mmtokenizer.eoa

        output_seq = None
        self.stage1_output_set = []

        sampling_params = SamplingParams(
            top_k=50,
            top_p=0.93,
            temperature=1.0,
            repetition_penalty=self.args.repetition_penalty,
            min_tokens=100,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.mmtokenizer.eoa],
            guidance_scale=1
        )

        #run_n_segments = min(self.args.run_n_segments, len(lyrics))+1
        run_n_segments = self.args.run_n_segments + 1
        #raw_output_list = []
        raw_output_list = [[] for _ in range(len(batch_configs))]
        for i in tqdm(range(run_n_segments), desc="Stage1 inference..."):
            if i == 0:
                continue
            prompt_ids_list = []
            input_ids_list = []
            for idx,config in enumerate(batch_configs):
                
                args_copy = self._create_args_copy(config)
        
                # 加载歌词和风格
                lyrics = self._load_lyrics(args_copy)
                genres = self._load_genres(args_copy)
        
                # 构建prompt
                prompt_texts = self._build_prompts(lyrics, genres)
                #audio_prompt = self._process_audio_prompt(args_copy)
        
                p = prompt_texts[i]

                # 构建当前segment的prompt
                section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
                if i == 1:
                    if args_copy.use_audio_prompt or args_copy.use_dual_tracks_prompt:
                        audio_prompt = self._process_audio_prompt(args_copy)
                        prompt_ids = (
                            self.mmtokenizer.tokenize(prompt_texts[0]) +
                            self.mmtokenizer.tokenize("[start_of_reference]") +
                            [soa_token] + self.codectool.sep_ids + audio_prompt + [eoa_token] +
                            self.mmtokenizer.tokenize("[end_of_reference]") +
                            start_of_segment + self.mmtokenizer.tokenize(section_text) +
                            [soa_token] + self.codectool.sep_ids
                        )
                    else:
                        prompt_ids = self.mmtokenizer.tokenize(prompt_texts[0]) + start_of_segment + self.mmtokenizer.tokenize(section_text) + [soa_token] + self.codectool.sep_ids
                else:
                    prompt_ids = end_of_segment + start_of_segment + self.mmtokenizer.tokenize(section_text) + [soa_token] + self.codectool.sep_ids
                
                prompt_ids_list.append(prompt_ids)
                # 上下文窗口管理
                max_context = 16384 - args_copy.max_new_tokens - 1
                input_ids = raw_output_list[idx] + prompt_ids if i > 1 else prompt_ids
                if len(input_ids) > max_context:
                    print(f'Section {i}: output length {len(input_ids)} exceeding context length {max_context}, now using the last {max_context} tokens.')
                    input_ids = input_ids[-max_context:]
                
                input_ids_list.append(input_ids)

            # 生成逻辑
            p = []
            for input_ids in input_ids_list:
                #input_ids_with_special_token = input_ids_list.append(32016)
                input_ids_with_special_token = input_ids + [32016]
                p.append(input_ids_with_special_token)
            if i == 1:
                sampling_params = SamplingParams(
                        top_k=50,
                        top_p=0.93,
                        temperature=1.0,
                        repetition_penalty=self.args.repetition_penalty,
                        min_tokens=100,
                        max_tokens=self.max_new_tokens,
                        stop_token_ids=[self.mmtokenizer.eoa],
                        guidance_scale=1.0
                    )
            else:
                sampling_params = SamplingParams(
                        top_k=50,
                        top_p=0.93,
                        temperature=1.0,
                        repetition_penalty=self.args.repetition_penalty,
                        min_tokens=100,
                        max_tokens=self.max_new_tokens,
                        stop_token_ids=[self.mmtokenizer.eoa],
                        guidance_scale=1.0
                    )
            with torch.no_grad():
                print(f"Start generating for segment {i}") 
                batch_output = self.stage1_model.generate(
                    prompt_token_ids=p,
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                print(f"Generation completed for {len(batch_output)} samples")
                output_seq_list = list(list(single_output.outputs[0].token_ids) for single_output in batch_output)
                for output_seq in output_seq_list:
                    if output_seq[-1] != eoa_token:
                        output_seq.append(eoa_token)

                if len(output_seq_list) != len(batch_configs):
                    raise ValueError(f"num of outputs: {len(output_seq_list)} doesn't match num of requests: {len(batch_configs)}")

            # 累积输出
            #raw_output_list = prompt_ids_list + output_seq_list if i == 1 else raw_output_list + prompt_ids_list + output_seq_list
            
            #for idx, output_seq in enumerate(output_seq_list):
            #    raw_output_list[idx] = prompt_ids_list[idx] + output_seq if i==1 else raw_output_list[idx] + prompt_ids_list[idx] + output_seq
        
            for idx in range(len(batch_configs)):
                current_prompt = prompt_ids_list[idx]
                current_output = output_seq_list[idx]
                if i > 1:
                    raw_output_list[idx] = raw_output_list[idx] + current_prompt + current_output
                else:
                    raw_output_list[idx] = current_prompt + current_output    

        # 解析生成的代码
        for idx, config in enumerate(tqdm(batch_configs, desc="Stage1 audio code analysis...")):
            args_copy = self._create_args_copy(config)
            
            stage1_output_dir = os.path.join(args_copy.output_dir, f"stage1")
            os.makedirs(stage1_output_dir, exist_ok=True)

            random_id = uuid.uuid4()
            genres = self._load_genres(args_copy)

            ids = np.array(raw_output_list[idx])
            soa_idx = np.where(ids == soa_token)[0]
            eoa_idx = np.where(ids == eoa_token)[0]

            if len(soa_idx)!=len(eoa_idx):
                raise ValueError(f'in request: {idx}, invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

            vocals, instrumentals = [], []
            range_begin = 1 if args_copy.use_audio_prompt or args_copy.use_dual_tracks_prompt else 0
            for i in range(range_begin, len(soa_idx)):
                codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
                if codec_ids[0] == 32016:  # 特殊token过滤
                    codec_ids = codec_ids[1:]
                codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]

                # 分离音轨
                vocal_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
                inst_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        
                vocals.append(vocal_ids)
                instrumentals.append(inst_ids)

            vocals = np.concatenate(vocals, axis=1)
            instrumentals = np.concatenate(instrumentals, axis=1)
            vocal_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tk{sampling_params.top_k}_tp{sampling_params.top_p}_T{sampling_params.temperature}_rp{sampling_params.repetition_penalty}_maxtk{self.args.max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
            inst_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tk{sampling_params.top_k}_tp{sampling_params.top_p}_T{sampling_params.temperature}_rp{sampling_params.repetition_penalty}_maxtk{self.args.max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
            np.save(vocal_save_path, vocals)
            np.save(inst_save_path, instrumentals)
            self.stage1_output_set.append(vocal_save_path)
            self.stage1_output_set.append(inst_save_path)
        
            outputs.append({
                "vocals": vocals,
                "instrumentals": instrumentals,
                "config": config,
                "vocal_save_path": vocal_save_path,
                "inst_save_path": inst_save_path
            })
        
        if not self.args.disable_offload_model:
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.stage1_model.llm_engine.model_executor
            del self.stage1_model
            torch.cuda.synchronize()
            #if torch.distributed.is_initialized():
            #    try:
            #        torch.distributed.destroy_process_group()
            #    except Exception as e:
            #        print(f"Cleanup warning: {str(e)}")
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            #ray.shutdown()
            print("Successfully delete the stage1 llm pipeline and free the GPU memory.")

        return outputs

    def _logits_processor_stage1(self, token_ids, logits):
        blocked_token_ids = list(range(0, 32002))+[32016]
        logits[blocked_token_ids] = -float("inf")
        return logits

    def _create_args_copy(self, config):
        args_copy = copy.deepcopy(self.args)
        for k, v in config.items():
            if hasattr(args_copy, k):
                setattr(args_copy, k, v)
        return args_copy

    def _load_lyrics(self, args):
        with open(args.lyrics_txt) as f:
            return split_lyrics(f.read())

    def _load_genres(self, args):
        with open(args.genre_txt) as f:
            return f.read().strip()

    def _build_prompts(self, lyrics, genres):
        full_lyrics = "\n".join(lyrics)
        return [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"] + lyrics

    def _interleave_tracks(self, vocals_ids, instrumental_ids, args):
        ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
        audio_prompt_codec = ids_segment_interleaved[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
        audio_prompt_codec = audio_prompt_codec.tolist()
        return audio_prompt_codec
    
    def _process_audio_prompt(self, args):
        if args.use_audio_prompt and not args.audio_prompt_path:
            raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
        if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
            raise FileNotFoundError("Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!")

        if args.use_dual_tracks_prompt:
            vocals = self._encode_audio(args.vocal_track_prompt_path, args)
            instrumental = self._encode_audio(args.instrumental_track_prompt_path, args)
            audio_prompt_codec = self._interleave_tracks(vocals, instrumental, args)
        elif args.use_audio_prompt:
            #return self._encode_single_audio(args)
            raw_codes = self._encode_audio(args.audio_prompt_path, args)
            code_ids = self.codectool.npy2ids(raw_codes[0])
            audio_prompt_codec = code_ids[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
        return audio_prompt_codec

    def _encode_audio(self, path, args, target_bw=0.5):
        audio_prompt = load_audio_mono(path)
        device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
        if len(audio_prompt.shape)<3:
            audio_prompt = audio_prompt.unsqueeze_(0)
        with torch.no_grad():
            raw_codes = self.codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int32)
        return self.codectool.npy2ids(raw_codes[0])

    def _process_stage2(self, stage1_results, batch_configs):
        print("Stage 2 inference...")
        
        self.stage2_model = LLM(
            model=self.args.stage2_model, 
            dtype=torch.bfloat16,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            max_model_len=8192,
            #disable_distributed_init=True,  # 显式禁用分布式初始化
            #worker_use_ray=False,  # 彻底禁用Ray
            enforce_eager=True,      # 禁用Ray异步执行
            tensor_parallel_size=1,  # 禁用分布式
            gpu_memory_utilization=0.9,   # 显存利用率调整
            disable_custom_all_reduce=True 
        )
        stage2_results = self._stage2_inference(stage1_results, batch_size=self.args.stage2_batch_size)
        print(stage2_results)
        print('Stage 2 DONE.\n')
        if not args.disable_offload_model:
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.stage2_model.llm_engine.model_executor
            del self.stage2_model
            torch.cuda.synchronize()
            #if torch.distributed.is_initialized():
            #    try:
            #        torch.distributed.destroy_process_group()
            #    except Exception as e:
            #        print(f"Cleanup warning: {str(e)}")         
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()   
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()            
            #ray.shutdown()
            print("Successfully delete the llm stage2 pipeline and free the GPU memory.")
        
        return stage2_results
        
    def _stage2_generate(self, prompt, batch_size=16):
        # process audio codes to tokens
        model = self.stage2_model
        codec_ids = self.codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codectool.offset_tok_ids(
                        codec_ids, 
                        global_offset=self.codectool.global_offset, 
                        codebook_size=self.codectool.codebook_size, 
                        num_codebooks=self.codectool.num_codebooks, 
                    ).astype(np.int32)
        # Prepare prompt_ids based on batch size or single input
        segments = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            segments.append(codec_ids[:, idx_begin:idx_end][0])
        segments_np = np.array(segments)
        # Initialize prompt array with numpy
        prompt_batch = np.zeros((batch_size, 300 + 3 + 300*8), dtype=np.int32)  # Pre-allocate
        prompt_lengths = np.zeros(batch_size, dtype=np.int32)
        # Initial prompt setup
        for i in range(batch_size):
            initial = [self.mmtokenizer.soa, self.mmtokenizer.stage_1] + segments[i].tolist() + [self.mmtokenizer.stage_2]
            prompt_batch[i, :len(initial)] = initial
            prompt_lengths[i] = len(initial)
        # stage2 smapling params
        def logits_processor_stage2(token_ids, logits):
            blocked_token_ids = list(range(0, 46358)) + list(range(53526, self.stage2_model.get_tokenizer().vocab_size))
            logits[blocked_token_ids] = -float("inf")
            return logits
    
        sampling_params = SamplingParams(
                min_tokens=7,
                max_tokens=7,
                stop_token_ids=[self.mmtokenizer.eoa],
                logits_processors=[logits_processor_stage2],
                ignore_eos=True,
                skip_special_tokens=False,
            )
        # Teacher forcing generate loop
        for frames_idx in tqdm(range(len(segments[0])), desc="Stage2 inference..."):
            # Vectorized append from segments
            current_tokens = segments_np[:, frames_idx]
            prompt_batch[np.arange(batch_size), prompt_lengths] = current_tokens
            prompt_lengths += 1
            # model generate
            prompt_list = [
                prompt_batch[i, :prompt_lengths[i]].tolist() 
                    for i in range(batch_size)
                ]
            batch_output = model.generate(
                prompt_token_ids=prompt_list,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            # Vectorized output processing
            generated_tokens = np.array([o.outputs[0].token_ids for o in batch_output], dtype=np.int32)
            row_idx = np.arange(batch_size)[:, None]  # (B,1)
            col_idx = prompt_lengths[:, None] + np.arange(7)  # (B,7)
            # Assign using advanced indexing
            prompt_batch[row_idx, col_idx] = generated_tokens
            prompt_lengths += 7

        # Extract final results
        output = np.concatenate([prompt_batch[i, len(initial):prompt_lengths[i]] for i in range(batch_size)])

        return output

    def _stage2_inference(self, stage1_results, batch_size=4):
        stage2_results = []
        for i,stage1_result in enumerate(tqdm(stage1_results, desc="Stage2 inference...")):
            args_copy = self._create_args_copy(stage1_result["config"])
            stage2_output_dir = os.path.join(args_copy.output_dir, 'stage2')
            os.makedirs(stage2_output_dir, exist_ok=True)
            output_filename_vocal = os.path.join(stage2_output_dir, os.path.basename(stage1_result["vocal_save_path"]))
            output_filename_inst = os.path.join(stage2_output_dir, os.path.basename(stage1_result["inst_save_path"]))
        
            self._stage2_inference_post(output_filename_vocal, stage1_result["vocal_save_path"], batch_size)
            self._stage2_inference_post(output_filename_inst, stage1_result["inst_save_path"], batch_size)
            stage2_results.append({
                "config": stage1_result["config"],
                "vocal_and_inst_save_path": [output_filename_vocal, output_filename_inst]
            })
        return stage2_results

    def _stage2_inference_post(self, output_filename, input_filename, batch_size):
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            return None
        
        # Load the prompt
        prompt = np.load(input_filename).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = self._stage2_generate(prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in tqdm(range(num_segments), desc="Stage2 inference..."):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = self._stage2_generate(
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)
        
        # Process the ending part of the prompt
        if output_duration*50 != prompt.shape[-1]:
            ending = self._stage2_generate(prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = self.codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        np.save(output_filename, fixed_output)
        return fixed_output
    def _stage2_logits_processor(self, logits):
        """Stage2 logits处理"""
        blocked_tokens = list(range(0, 46358)) + list(range(53526, self.stage2_model.get_tokenizer().vocab_size))
        logits[:, blocked_tokens] = -float('inf')
        return logits
    def _process_audio_outputs(self, stage2_results):
        # reconstruct tracks
        #recons_output_dir = os.path.join(args.output_dir, "recons")
        #recons_mix_dir = os.path.join(recons_output_dir, 'mix')
        #os.makedirs(recons_mix_dir, exist_ok=True)
        #tracks = []
        for stage2_result in stage2_results:
            args_copy = self._create_args_copy(stage2_result["config"])
            recons_output_dir = os.path.join(args_copy.output_dir, "recons")
            recons_mix_dir = os.path.join(recons_output_dir, 'mix')
            os.makedirs(recons_mix_dir, exist_ok=True)
            tracks = []
            for npy in stage2_result["vocal_and_inst_save_path"]:
                codec_result = np.load(npy)
                decodec_rlt=[]
                with torch.no_grad():
                    decoded_waveform = self.codec_model.decode(torch.as_tensor(codec_result.astype(np.int32), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(self.device))
                decoded_waveform = decoded_waveform.cpu().squeeze(0)
                decodec_rlt.append(torch.as_tensor(decoded_waveform))
                decodec_rlt = torch.cat(decodec_rlt, dim=-1)
                save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
                tracks.append(save_path)
                save_audio(decodec_rlt, save_path, 16000)
            # mix tracks
            for inst_path in tracks:
                try:
                    if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                        and '_itrack' in inst_path:
                        # find pair
                        vocal_path = inst_path.replace('_itrack', '_vtrack')
                        if not os.path.exists(vocal_path):
                            continue
                        # mix
                        recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
                        vocal_stem, sr = sf.read(inst_path)
                        instrumental_stem, _ = sf.read(vocal_path)
                        mix_stem = (vocal_stem + instrumental_stem) / 1
                        sf.write(recons_mix, mix_stem, sr)
                except Exception as e:
                    print(e)

            # vocoder to upsample audios
            vocal_decoder, inst_decoder = build_codec_model(self.args.config_path, self.args.vocal_decoder_path, self.args.inst_decoder_path)
            vocoder_output_dir = os.path.join(args_copy.output_dir, 'vocoder')
            vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
            vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
            os.makedirs(vocoder_mix_dir, exist_ok=True)
            os.makedirs(vocoder_stems_dir, exist_ok=True)
            for npy in stage2_result["vocal_and_inst_save_path"]:
                if '_itrack' in npy:
                    # Process instrumental
                    instrumental_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'itrack.mp3'),
                        self.args.rescale,
                        args_copy,
                        inst_decoder,
                        self.codec_model
                    )
                else:
                    # Process vocal
                    vocal_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
                        self.args.rescale,
                        args_copy,
                        vocal_decoder,
                        self.codec_model
                    )
            # mix tracks
            try:
                mix_output = instrumental_output + vocal_output
                vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
                save_audio(mix_output, vocoder_mix, 44100, self.args.rescale)
                print(f"Created mix: {vocoder_mix}")
            except RuntimeError as e:
                print(e)
                print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

            # Post process
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,     # 16kHz
                b_file=vocoder_mix,     # 48kHz
                c_file=os.path.join(args_copy.output_dir, os.path.basename(recons_mix)),
                cutoff_freq=5500.0
            )
def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

def load_audio_mono(filepath, sr=16000):
    audio, orig_sr = torchaudio.load(filepath)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if orig_sr != sr:
        resample = Resample(orig_freq=sr, new_freq=sr)
        audio = resample(audio)
    return audio

    # convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

#region 主流程
def main():
    seed_everything(args.seed)
    
    # 加载所有配置
    configs = []
    for fname in os.listdir(args.prompts_config_path):
        if fname.endswith('.json'):
            with open(os.path.join(args.prompts_config_path, fname)) as f:
                configs.append(json.load(f))
    
    # 初始化处理器
    processor = BatchProcessor(args)
    
    # 分批处理
    for i in range(0, len(configs), args.request_batch_size):
        batch = configs[i:i+args.request_batch_size] if i+args.request_batch_size < len(configs) else configs[i:]
        processor.process_batch(batch)

if __name__ == "__main__":
    main()
#endregion