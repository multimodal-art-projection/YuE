
import os
import sys
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
# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
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


parser = argparse.ArgumentParser()
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
if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
    raise FileNotFoundError("Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!")
stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, f"stage1")
stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)


def seed_everything(seed=42): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(args.seed)
# load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
# fp32 is neccessary to align with generation by huggingface transformers
# model = LLM(model=stage1_model, dtype=torch.float32)
# model_stage2 = LLM(
#     model=stage2_model, 
#     dtype=torch.float32
#     )
# import pdb; pdb.set_trace()
model = LLM(model=stage1_model, 
            dtype=torch.bfloat16)

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def logits_processor_stage1(token_ids, logits):
    blocked_token_ids = list(range(0, 32002))+[32016]
    logits[blocked_token_ids] = -float("inf")
    return logits

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    # raw_codes = raw_codes.cpu().numpy().astype(np.int32)
    raw_codes = raw_codes.cpu().numpy().astype(np.int32)
    return raw_codes

def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics

# Call the function and print the result
stage1_output_set = []
# Tips:
# genre tags support instrumental，genre，mood，vocal timbr and vocal gender
# all kinds of tags are needed
with open(args.genre_txt) as f:
    genres = f.read().strip()
with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())
# intruction
full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics


random_id = uuid.uuid4()
output_seq = None
# Here is suggested decoding config
# TODO: implement classifier-free guidance 
# sampling_params = SamplingParams(
#             top_k=50,
#             top_p=0.93,
#             temperature=1.0,
#             repetition_penalty=args.repetition_penalty,
#             min_tokens=100,
#             max_tokens=max_new_tokens,
#             stop_token_ids=[mmtokenizer.eoa],
#             logits_processors=[logits_processor_stage1]
#         )

sampling_params = SamplingParams(
            top_k=50,
            top_p=0.93,
            temperature=1.0,
            repetition_penalty=args.repetition_penalty,
            min_tokens=100,
            max_tokens=max_new_tokens,
            stop_token_ids=[mmtokenizer.eoa],
            guidance_scale=1.5
        )

# special tokens
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
# Format text prompt
run_n_segments = min(args.run_n_segments, len(lyrics))+1

for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    if i==0:
        continue
    if i==1:
        if args.use_dual_tracks_prompt or args.use_audio_prompt:
            if args.use_dual_tracks_prompt:
                vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                instrumental_ids = load_audio_mono(args.instrumental_track_prompt_path)
                vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                vocals_ids = codectool.npy2ids(vocals_ids[0])
                instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
                audio_prompt_codec = ids_segment_interleaved[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
                audio_prompt_codec = audio_prompt_codec.tolist()
            elif args.use_audio_prompt:
                audio_prompt = load_audio_mono(args.audio_prompt_path)
                raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                # Format audio prompt
                code_ids = codectool.npy2ids(raw_codes[0])
                audio_prompt_codec = code_ids[int(args.prompt_start_time *50): int(args.prompt_end_time *50)] # 50 is tps of xcodec
            audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
            sentence_ids = mmtokenizer.tokenize("[start_of_reference]") +  audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
            head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
        else:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
        prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    else:
        prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

    input_ids = raw_output + prompt_ids if i > 1 else prompt_ids

    # Use window slicing in case output sequence exceeds the context of model
    max_context = 16384-max_new_tokens-1
    if len(input_ids) > max_context:
        print(f'Section {i}: output length {len(input_ids)} exceeding context length {max_context}, now using the last {max_context} tokens.')
        input_ids = input_ids[-(max_context):]
    # import pdb; pdb.set_trace()
    with torch.no_grad():

        batch_output = model.generate(
            prompt_token_ids=[input_ids, [32016]],
            sampling_params=sampling_params,
            use_tqdm=False
            )
        # import pdb; pdb.set_trace()
        output_seq = list(batch_output[0].outputs[0].token_ids)
        if output_seq[-1] != mmtokenizer.eoa:
            output_seq.append(mmtokenizer.eoa)
    if i > 1:
        raw_output = raw_output + prompt_ids + output_seq
    else:
        raw_output = prompt_ids + output_seq

# save raw output and check sanity
ids = np.array(raw_output)
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
if len(soa_idx)!=len(eoa_idx):
    raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

vocals = []
instrumentals = []
range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0
for i in range(range_begin, len(soa_idx)):
    codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
    if codec_ids[0] == 32016:
        codec_ids = codec_ids[1:]
    codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
    vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
    vocals.append(vocals_ids)
    instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
    instrumentals.append(instrumentals_ids)
vocals = np.concatenate(vocals, axis=1)
instrumentals = np.concatenate(instrumentals, axis=1)
vocal_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tk{sampling_params.top_k}_tp{sampling_params.top_p}_T{sampling_params.temperature}_rp{sampling_params.repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
inst_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tk{sampling_params.top_k}_tp{sampling_params.top_p}_T{sampling_params.temperature}_rp{sampling_params.repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)
stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)

# import pdb; pdb.set_trace()
# offload stage1 model
if not args.disable_offload_model:
    destroy_model_parallel()
    destroy_distributed_environment()
    del model.llm_engine.model_executor
    del model
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")

print("Stage 2 inference...")
model_stage2 = LLM(
    model=stage2_model, 
    dtype=torch.bfloat16
    )

def stage2_generate(model, prompt, batch_size=16):
    # process audio codes to tokens
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
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
        initial = [mmtokenizer.soa, mmtokenizer.stage_1] + segments[i].tolist() + [mmtokenizer.stage_2]
        prompt_batch[i, :len(initial)] = initial
        prompt_lengths[i] = len(initial)
    # stage2 smapling params
    def logits_processor_stage2(token_ids, logits):
        blocked_token_ids = list(range(0, 46358)) + list(range(53526, model_stage2.get_tokenizer().vocab_size))
        logits[blocked_token_ids] = -float("inf")
        return logits
    
    sampling_params = SamplingParams(
            min_tokens=7,
            max_tokens=7,
            stop_token_ids=[mmtokenizer.eoa],
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

def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            continue
        
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in tqdm(range(num_segments), desc="Stage2 inference..."):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)
        
        # Process the ending part of the prompt
        if output_duration*50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

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
        stage2_result.append(output_filename)
    return stage2_result

stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
print(stage2_result)
print('Stage 2 DONE.\n')
if not args.disable_offload_model:
    destroy_model_parallel()
    destroy_distributed_environment()
    del model_stage2.llm_engine.model_executor
    del model_stage2
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm stage2 pipeline and free the GPU memory.")
# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
# reconstruct tracks
recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, 'mix')
os.makedirs(recons_mix_dir, exist_ok=True)
tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    decodec_rlt=[]
    with torch.no_grad():
        decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int32), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
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
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)
for npy in stage2_result:
    if '_itrack' in npy:
        # Process instrumental
        instrumental_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'itrack.mp3'),
            args.rescale,
            args,
            inst_decoder,
            codec_model
        )
    else:
        # Process vocal
        vocal_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
            args.rescale,
            args,
            vocal_decoder,
            codec_model
        )
# mix tracks
try:
    mix_output = instrumental_output + vocal_output
    vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
    save_audio(mix_output, vocoder_mix, 44100, args.rescale)
    print(f"Created mix: {vocoder_mix}")
except RuntimeError as e:
    print(e)
    print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

# Post process
replace_low_freq_with_energy_matched(
    a_file=recons_mix,     # 16kHz
    b_file=vocoder_mix,     # 48kHz
    c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
    cutoff_freq=5500.0
)
