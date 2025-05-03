import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import re
import random
import uuid
import copy
import json
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

class RequestBatchProcessor:
    def __init__(self, base_args):
        self.base_args = base_args
        self.codec_model = None
        self.model_stage1 = None
        self.model_stage2 = None
        self.mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)
        self.device = torch.device(f"cuda:{base_args.cuda_idx}" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Lazily load models when needed"""
        if self.model_stage1 is None:
            self.model_stage1 = LLM(
                model=self.base_args.stage1_model,
                dtype=torch.bfloat16,
                tensor_parallel_size=torch.cuda.device_count()
            )
            
        if self.codec_model is None:
            model_config = OmegaConf.load(self.base_args.basic_model_config)
            self.codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(self.device)
            parameter_dict = torch.load(self.base_args.resume_path, map_location='cpu', weights_only=False)
            self.codec_model.load_state_dict(parameter_dict['codec_model'])
            self.codec_model.to(self.device)
            self.codec_model.eval()

    def process_batch(self, requests):
        """Process a batch of requests"""
        batch_prompts = []
        batch_sampling_params = []
        batch_metadata = []
        
        for req in requests:
            current_args = self._prepare_request_args(req)
            prompts, metadata = self._prepare_prompts(current_args)
            batch_prompts.extend(prompts)
            batch_sampling_params.extend([
                self._create_sampling_params(current_args, i) 
                for i in range(len(prompts))
            ])
            batch_metadata.append(metadata)
            
        outputs = self.model_stage1.generate(
            prompts=batch_prompts,
            sampling_params=batch_sampling_params,
            use_tqdm=False
        )
        
        return self._process_outputs(outputs, batch_metadata)

    def _prepare_request_args(self, req_config):
        """Create per-request arguments with base config override"""
        args = copy.deepcopy(self.base_args)
        
        # Validate required parameters
        required_fields = ['genre_txt', 'lyrics_txt', 'output_dir']
        for field in required_fields:
            if field not in req_config:
                raise ValueError(f"Missing required field in config: {field}")
                
        # Update arguments
        args.__dict__.update(req_config)
        
        # Validate file paths
        self._validate_paths(args)
        return args

    def _validate_paths(self, args):
        """Validate all file paths in arguments"""
        if not os.path.exists(args.genre_txt):
            raise FileNotFoundError(f"Genre file not found: {args.genre_txt}")
        if not os.path.exists(args.lyrics_txt):
            raise FileNotFoundError(f"Lyrics file not found: {args.lyrics_txt}")
        if args.use_audio_prompt and not os.path.exists(args.audio_prompt_path):
            raise FileNotFoundError(f"Audio prompt not found: {args.audio_prompt_path}")
        if args.use_dual_tracks_prompt:
            if not os.path.exists(args.vocal_track_prompt_path):
                raise FileNotFoundError(f"Vocal track not found: {args.vocal_track_prompt_path}")
            if not os.path.exists(args.instrumental_track_prompt_path):
                raise FileNotFoundError(f"Instrumental track not found: {args.instrumental_track_prompt_path}")

    def _prepare_prompts(self, args):
        """Prepare prompts for a single request"""
        with open(args.genre_txt) as f:
            genres = f.read().strip()
        with open(args.lyrics_txt) as f:
            lyrics = self.split_lyrics(f.read())
            
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [
            f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}",
            *lyrics
        ]
        
        metadata = {
            'genres': genres,
            'lyrics': lyrics,
            'output_dir': args.output_dir,
            'request_id': str(uuid.uuid4()),
            'audio_prompt': self._process_audio_prompt(args),
            'sampling_params': {
                'top_k': args.top_k,
                'top_p': args.top_p,
                'temperature': args.temperature,
                'repetition_penalty': args.repetition_penalty
            }
        }
        
        return prompt_texts[:min(args.run_n_segments, len(lyrics))+1], metadata

    def _process_audio_prompt(self, args):
        """Process audio prompt for current request"""
        if args.use_dual_tracks_prompt:
            vocals = self.load_audio(args.vocal_track_prompt_path)
            instrumental = self.load_audio(args.instrumental_track_prompt_path)
            return self.encode_dual_tracks(vocals, instrumental, args)
        elif args.use_audio_prompt:
            audio = self.load_audio(args.audio_prompt_path)
            return self.encode_single_track(audio, args)
        return None

    def _create_sampling_params(self, args, segment_idx):
        """Create sampling params with CFG support"""
        return SamplingParams(
            top_k=50,
            top_p=0.93,
            temperature=1.0,
            repetition_penalty=args.repetition_penalty,
            min_tokens=100,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[self.mmtokenizer.eoa],
            guidance_scale=1.5 if segment_idx == 0 else 1.2,
            logits_processors=[self.logits_processor_stage1]
        )

    def logits_processor_stage1(self, token_ids, logits):
        """Custom logits processor for stage1"""
        blocked_token_ids = list(range(0, 32002)) + [32016]
        logits[:, blocked_token_ids] = -float("inf")
        return logits

    def _process_outputs(self, outputs, batch_metadata):
        """Process batch outputs into final results"""
        results = []
        output_ptr = 0
        
        for meta in batch_metadata:
            num_prompts = len(meta['prompts'])
            req_outputs = outputs[output_ptr:output_ptr+num_prompts]
            output_ptr += num_prompts
            
            processed = self._process_single_output(req_outputs, meta)
            results.append(processed)
            
        return results

    def _process_single_output(self, outputs, metadata):
        """Process outputs for a single request"""
        raw_output = []
        for i, output in enumerate(outputs):
            output_seq = list(output.outputs[0].token_ids)
            if output_seq[-1] != self.mmtokenizer.eoa:
                output_seq.append(self.mmtokenizer.eoa)
            raw_output.extend(output_seq)
            
        ids = np.array(raw_output)
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"Mismatched SOA/EOA pairs: {len(soa_idx)} vs {len(eoa_idx)}")
            
        vocals, instrumentals = self.extract_tracks(ids, soa_idx, eoa_idx, metadata)
        
        vocal_path, inst_path = self.save_tracks(
            vocals, instrumentals, 
            metadata['output_dir'], 
            metadata['request_id']
        )
        
        return {
            'vocals_path': vocal_path,
            'instruments_path': inst_path,
            'metadata': metadata
        }

    def extract_tracks(self, ids, soa_idx, eoa_idx, metadata):
        """Extract vocal and instrumental tracks from codes"""
        range_begin = 1 if metadata['audio_prompt'] else 0
        vocals = []
        instrumentals = []
        
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
                
            codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]
            vocals_ids = self.codectool.ids2npy(
                rearrange(codec_ids, "(n b) -> b n", b=2)[0]
            )
            instrumentals_ids = self.codectool.ids2npy(
                rearrange(codec_ids, "(n b) -> b n", b=2)[1]
            )
            vocals.append(vocals_ids)
            instrumentals.append(instrumentals_ids)
            
        return np.concatenate(vocals, axis=1), np.concatenate(instrumentals, axis=1)

    def save_tracks(self, vocals, instrumentals, output_dir, request_id):
        """Save generated tracks to disk"""
        os.makedirs(output_dir, exist_ok=True)
        vocal_path = os.path.join(
            output_dir,
            f"vocal_{request_id}.npy"
        )
        inst_path = os.path.join(
            output_dir,
            f"instrumental_{request_id}.npy"
        )
        np.save(vocal_path, vocals)
        np.save(inst_path, instrumentals)
        return vocal_path, inst_path

    # Audio processing utilities
    @staticmethod
    def split_lyrics(lyrics):
        """Split lyrics into structured segments"""
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

    def load_audio(self, path, sr=16000):
        """Load and resample audio file"""
        audio, orig_sr = torchaudio.load(path)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if orig_sr != sr:
            resampler = Resample(orig_sr, sr)
            audio = resampler(audio)
        return audio

    def encode_audio(self, audio, target_bw=0.5):
        """Encode audio to codec tokens"""
        with torch.no_grad():
            raw_codes = self.codec_model.encode(
                audio.to(self.device), 
                target_bw=target_bw
            )
        return raw_codes.transpose(0, 1).cpu().numpy().astype(np.int32)

    def encode_single_track(self, audio, args):
        """Encode single track audio prompt"""
        raw_codes = self.encode_audio(audio)
        code_ids = self.codectool.npy2ids(raw_codes[0])
        return code_ids[
            int(args.prompt_start_time * 50): 
            int(args.prompt_end_time * 50)
        ]

    def encode_dual_tracks(self, vocal, instrumental, args):
        """Encode dual track audio prompt"""
        vocal_codes = self.encode_audio(vocal)
        inst_codes = self.encode_audio(instrumental)
        vocal_ids = self.codectool.npy2ids(vocal_codes[0])
        inst_ids = self.codectool.npy2ids(inst_codes[0])
        interleaved = rearrange(
            [np.array(vocal_ids), np.array(inst_ids)], 
            'b n -> (n b)'
        )
        return interleaved[
            int(args.prompt_start_time*50*2): 
            int(args.prompt_end_time*50*2)
        ]

    def stage2_inference(self, model, stage1_outputs, output_dir, batch_size):
        """Batch process stage2 inference with multiple requests support"""
        stage2_result = []
        for output_file in tqdm(stage1_outputs, desc="Processing Stage2"):
            if not os.path.exists(output_file):
                print(f"Skipping missing file: {output_file}")
                continue
            
            request_id = os.path.basename(output_file).split('_')[-2]
            request_output_dir = os.path.join(output_dir, request_id)
            os.makedirs(request_output_dir, exist_ok=True)
        
            try:
                prompt = np.load(output_file).astype(np.int32)
                output_duration = prompt.shape[-1] // 50 // 6 * 6
                num_batch = output_duration // 6
            
                # 处理完整批次
                if num_batch <= batch_size:
                    output = self._process_single_batch(model, prompt, num_batch)
                else:
                    output = self._process_large_batch(model, prompt, batch_size, num_batch, output_duration)
            
                # 处理剩余帧
                if output_duration*50 < prompt.shape[-1]:
                    ending = self._process_ending(model, prompt, output_duration)
                    output = np.concatenate([output, ending], axis=0)
            
                # 后处理并保存
                processed_output = self._post_process_output(output, request_output_dir)
                stage2_result.append(processed_output)
            
            except Exception as e:
                print(f"Error processing {output_file}: {str(e)}")
            
        return stage2_result

    def _process_large_batch(self, model, prompt, batch_size, num_batch, output_duration):
        """处理超过批大小的请求"""
        segments = []
        for seg in tqdm(range(0, num_batch, batch_size), desc="Processing Large Batch"):
            start = seg * batch_size * 300
            end = min((seg + batch_size) * 300, output_duration*50)
            segment = self._process_single_batch(
                model,
                prompt[:, start:end],
                min(batch_size, num_batch-seg)
            )
            segments.append(segment)
        return np.concatenate(segments, axis=0)

    def _process_ending(self, model, prompt, output_duration):
        """处理最后不满6秒的片段"""
        ending = prompt[:, output_duration*50:]
        if ending.shape[-1] > 0:  # 确保有剩余内容需要处理
            return self._process_single_batch(model, ending, 1)
        return np.array([])

    def _post_process_output(self, output, output_dir):
        """后处理并保存输出"""
        output = self.codectool_stage2.ids2npy(output)
    
        # 修复无效代码
        fixed_output = copy.deepcopy(output)
        for i in range(fixed_output.shape[0]):
            for j in range(fixed_output.shape[1]):
                if fixed_output[i,j] < 0 or fixed_output[i,j] > 1023:
                    counter = Counter(fixed_output[i])
                    most_common = counter.most_common(1)[0][0]
                    fixed_output[i,j] = most_common
                
        output_path = os.path.join(output_dir, "stage2_output.npy")
        np.save(output_path, fixed_output)
        return output_path

    def process_final_outputs(self, results, args):
        """批量处理最终音频输出"""
        for result in tqdm(results, desc="生成最终音频"):
            try:
                request_id = result['metadata']['request_id']
            
                # 初始化解码器
                vocal_decoder, inst_decoder = build_codec_model(
                    args.config_path,
                    args.vocal_decoder_path,
                    args.inst_decoder_path
                )
            
                # 创建输出目录
                recons_dir = os.path.join(result['metadata']['output_dir'], 'recons', request_id)
                vocoder_dir = os.path.join(result['metadata']['output_dir'], 'vocoder', request_id)
                os.makedirs(recons_dir, exist_ok=True)
                os.makedirs(vocoder_dir, exist_ok=True)
            
                # 处理每个音轨
                track_data = {'vocals': None, 'instruments': None}
                for track_type in ['vocals', 'instruments']:
                    npy_path = result[f"{track_type}_path"]
                    if not os.path.exists(npy_path):
                        continue
                
                    # 解码神经网络
                    codec_result = np.load(npy_path)
                    decoded = self.codec_model.decode(
                        torch.LongTensor(codec_result).unsqueeze(0).permute(1, 0, 2).to(self.device)
                    decoded = decoded.cpu().squeeze(0)
                
                    # 保存重建音频
                    recon_path = os.path.join(recons_dir, f"{track_type}.wav")
                    self.save_audio(decoded, recon_path, 16000, args.rescale)
                
                    # 声码器处理
                    vocoder_path = os.path.join(vocoder_dir, f"{track_type}.wav")
                    process_audio(
                        npy_path,
                        vocoder_path,
                        args.rescale,
                        args,
                        vocal_decoder if track_type == 'vocals' else inst_decoder,
                        self.codec_model
                    )
                    track_data[track_type] = vocoder_path
            
                # 混音和后处理
                if all(track_data.values()):
                    self.mix_and_postprocess(track_data, vocoder_dir, request_id)
    
            except Exception as e:
                print(f"请求{request_id}处理失败: {str(e)}")

    def mix_and_postprocess(self, track_data, output_dir, request_id):
        """混音和后处理"""
        try:
            # 混音
            vocal, sr = sf.read(track_data['vocals'])
            inst, _ = sf.read(track_data['instruments'])
            mix = (vocal + inst) / 2
            mix_path = os.path.join(output_dir, f"mix_{request_id}.wav")
            sf.write(mix_path, mix, sr)
        
            # 后期处理
            final_path = os.path.join(output_dir, f"final_{request_id}.wav")
            replace_low_freq_with_energy_matched(
                track_data['vocals'],
                mix_path,
                final_path,
                cutoff_freq=5500.0
            )
        except Exception as e:
            print(f"混音失败 {request_id}: {str(e)}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument("--stage1_model", type=str, required=True)
    parser.add_argument("--stage2_model", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--run_n_segments", type=int, default=2)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    
    # Batch Processing
    parser.add_argument("--request_batch_size", type=int, default=8)
    parser.add_argument("--prompts_config_path", type=str, required=True)
    
    # Hardware
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument("--disable_offload_model", action="store_true")
    
    # Codec Configuration
    parser.add_argument('--basic_model_config', required=True)
    parser.add_argument('--resume_path', required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--vocal_decoder_path', type=str, required=True)
    parser.add_argument('--inst_decoder_path', type=str, required=True)
    parser.add_argument('-r', '--rescale', action='store_true')
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    
    base_args = parser.parse_args()
    
    # Load request configs
    request_configs = []
    for cfg_file in os.listdir(base_args.prompts_config_path):
        if cfg_file.endswith('.json'):
            with open(os.path.join(base_args.prompts_config_path, cfg_file)) as f:
                req = json.load(f)
                request_configs.append(req)
                
    # Process requests
    processor = RequestBatchProcessor(base_args)
    results = []
    
    for i in range(0, len(request_configs), base_args.request_batch_size):
        batch = request_configs[i:i+base_args.request_batch_size]
        processor.load_models()
        batch_results = processor.process_batch(batch)
        results.extend(batch_results)
        
        if not base_args.disable_offload_model:
            destroy_model_parallel()
            destroy_distributed_environment()
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
            
    # Stage 2 Processing
    processor.model_stage2 = LLM(
        model=base_args.stage2_model,
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count()
    )
    
    for result in results:
        processor.stage2_inference(
            processor.model_stage2,
            [result['vocals_path'], result['instruments_path']],
            result['metadata']['output_dir'],
            base_args.stage2_batch_size
        )
        
    # Final audio processing
    processor.process_final_outputs(results, base_args)

if __name__ == "__main__":
    main()  