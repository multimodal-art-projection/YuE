# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse
import math
import json
import random
import numpy as np
import time
import gzip
import glob
import torch
import multiprocessing
import nltk
import einops
from einops import rearrange
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from core.tokenizer.mmtokenizer import _MMSentencePieceTokenizer
from core.datasets import indexed_dataset
# Assuming these functions exist in the specified module
from preprocess_data_conditional_xcodec import get_file_name, check_files_exist
from preprocess_data_conditional_xcodec import Encoder as EncoderBase

DEBUG = False
np.random.seed(42)

def get_size_in_bytes(arr):
    """Returns the size of a numpy array in bytes."""
    return arr.nbytes

def inverse_transform_sampling(cdf, x_values, num_samples=1):
    """
    Performs inverse transform sampling given a CDF.
    Used for sampling audio prompt lengths in ICL mode.
    """
    # Generate uniformly distributed random numbers
    r = np.random.rand(num_samples)
    # Find corresponding x values using interpolation on the CDF
    random_samples = np.interp(r, cdf, x_values)
    return random_samples

class Encoder(EncoderBase):
    """
    Encodes JSON lines into token IDs for different preprocessing modes.
    Handles text, codec, token-level interleaving, CoT, and ICL.
    """
    # Placeholders for CDF values used in inverse_transform_sampling for ICL.
    # These should ideally be loaded from configuration or data.

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def encode_mix_text_and_codec(self, json_line):
        """Encodes text and codec data, simple concatenation based on order."""
        data = json.loads(json_line)
        assert 'text' in data and 'codec' in data, "`text` and `codec` must be in the json key"

        ids = {}
        lens = {}

        segmented_lyrics = data['splitted_lyrics']['segmented_lyrics']
        raw_codec = np.load(data['codec'])

        full_length_of_song = data['audio_length_in_sec']
        # Handle potential division by zero or invalid full_length_of_song
        if full_length_of_song <= 0:
             print(f"Warning: Invalid audio_length_in_sec={full_length_of_song} in {data.get('id', 'unknown')}. Skipping.")
             return {}, {}, 0 # Return empty results and 0 bytes processed
        fps = raw_codec.shape[1] / full_length_of_song

        doc_ids = []
        sentence_lens = [] # here sentence means segment
        for segment in segmented_lyrics:
            duration = segment['duration']
            # Relaxed fps check allowing exactly 50.0
            # if fps > 51 or fps < 49:
            #     if DEBUG: print(f"fps={fps} is invalid, skipping...")
            #     if DEBUG: print(f"full_length_of_song={full_length_of_song}, raw_codec.shape[1]={raw_codec.shape[1]}")
            #     continue

            if duration <= 0 or duration > full_length_of_song:
                if DEBUG: print(f"duration={duration} is invalid, skipping...")
                continue
            # Check frame indices validity
            if not (0 <= segment['codec_frame_start'] < segment['codec_frame_end'] <= raw_codec.shape[1]):
                 if DEBUG: print(f"Invalid frame indices: start={segment['codec_frame_start']}, end={segment['codec_frame_end']}, total={raw_codec.shape[1]}. Skipping.")
                 continue
            # Check minimum frame length (ensure it's at least 1 frame, fps check handles very short)
            if segment['codec_frame_end'] - segment['codec_frame_start'] <= 0: # Stricter check: must be > 0
                 if DEBUG: print(f"Frame length is zero or negative: {segment['codec_frame_end'] - segment['codec_frame_start']}. Skipping.")
                 continue
            # Check if frame length is less than 1 second equivalent (fps frames)
            if segment['codec_frame_end'] - segment['codec_frame_start'] < fps:
                if DEBUG: print(f"frame too short: frame_end - frame_start={segment['codec_frame_end'] - segment['codec_frame_start']} (< {fps}), segment={segment}, skipping...")
                continue

            line_content = segment['line_content']
            raw_codec_segment = raw_codec[:, segment['codec_frame_start']:segment['codec_frame_end']]

            # tokenize the text
            instruction = self.args.instruction
            text = instruction + '\n' + line_content # Fixed newline escape

            if self.args.instruction_dropout_rate > 0.0:
                if np.random.rand() < self.args.instruction_dropout_rate:
                    text = line_content

            text_ids = Encoder.tokenizer.tokenize(text)

            # read codec npy
            try:
                codec_ids = [Encoder.tokenizer.soa] + Encoder.codectool.sep_ids + Encoder.codectool.npy2ids(raw_codec_segment) + [Encoder.tokenizer.eoa]

                if self.args.order == "textfirst":
                    sentence_ids = text_ids + codec_ids
                elif self.args.order == "audiofirst":
                    sentence_ids = codec_ids + text_ids
                else:
                    # Fallback or error if order is not textfirst/audiofirst for this function
                    print(f"Warning: Unexpected order '{self.args.order}' for encode_mix_text_and_codec. Defaulting to audiofirst.")
                    sentence_ids = codec_ids + text_ids


                doc_ids.extend(sentence_ids)
                sentence_lens.append(len(sentence_ids))
            except Exception as e:
                print(f"Error processing segment in encode_mix_text_and_codec: {e}")
                print(f"Data ID: {data.get('id', 'unknown')}, Codec Path: {data.get('codec', 'unknown')}")
                print(f"Segment: {segment}")
                print(f"Raw Codec Shape: {raw_codec.shape}")
                print(f"Frame Indices: start={segment['codec_frame_start']}, end={segment['codec_frame_end']}")
                print(f"Song Length: {full_length_of_song}, Calculated FPS: {fps}")
                print(f"Segment Codec Shape: {raw_codec_segment.shape}")
                print(f"Line Content: {line_content}")
                print(f"Text Input: {text}")


        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
            sentence_lens[-1] += 1

        key = "text" # hardcode key
        ids[key] = doc_ids
        lens[key] = sentence_lens

        # Estimate size processed, handle case where raw_codec might not exist if skipped early
        bytes_processed = len(json_line)
        if 'raw_codec' in locals() and isinstance(raw_codec, np.ndarray):
             bytes_processed += get_size_in_bytes(raw_codec)

        return ids, lens, bytes_processed


    def encode_codec_stage_2(self, json_line):
        """Encodes codec data for stage 2 training."""
        data = json.loads(json_line)

        ids = {}
        lens = {}

        raw_codec = np.load(data[Encoder.codectool.data_feature]).astype(np.int32)
        raw_codec = torch.as_tensor(raw_codec, dtype=torch.int32)
        # fps*duration: 50fps*6s = 300
        fps = Encoder.codectool.fps
        duration = 6 # Target duration for stage 2 segments
        segment_length = fps * duration

        # Ensure raw_codec has a temporal dimension before splitting
        if raw_codec.ndim < 2 or raw_codec.shape[1] == 0:
            print(f"Warning: Invalid raw_codec shape {raw_codec.shape} for stage 2 in {data.get('id', 'unknown')}. Skipping.")
            return {}, {}, len(json_line) + get_size_in_bytes(raw_codec)

        segmented_frames_all = torch.split(raw_codec, segment_length, dim=1)

        # Keep only segments that have the exact length (discard last partial segment)
        segmented_frames_all = [frame for frame in segmented_frames_all if frame.shape[1] == segment_length]

        doc_ids = []
        sentence_lens = [] # here sentence means segment
        for frames in segmented_frames_all:
            try:
                # extract specified layers of codebooks
                quantizer_begin = Encoder.codectool.quantizer_begin
                n_quantizer = Encoder.codectool.n_quantizer
                codes = frames[quantizer_begin : quantizer_begin + n_quantizer].numpy()

                # convert codes to ids
                flattened_ids = np.array(Encoder.codectool.npy2ids(codes))
                # Check if flattened_ids is empty, which can happen if npy2ids fails or codes are invalid
                if flattened_ids.size == 0:
                     print(f"Warning: flattened_ids is empty for a segment in {data.get('id', 'unknown')}. Skipping segment.")
                     continue

                unflattened_ids = Encoder.codectool.unflatten(flattened_ids, n_quantizer)
                # Check dimensions after unflattening
                if unflattened_ids.shape[0] == 0 or unflattened_ids.shape[1] == 0:
                     print(f"Warning: unflattened_ids has zero dimension {unflattened_ids.shape} in {data.get('id', 'unknown')}. Skipping segment.")
                     continue

                codebook_0 = unflattened_ids[0]
                # count num of unique codes, if < 25, skip (ensure enough variation)
                if len(np.unique(codebook_0)) < 25:
                    continue

                codebook_rest = unflattened_ids[1:]
                codebook_0_list = codebook_0.tolist()
                codebook_rest_list = einops.rearrange(codebook_rest, 'K T -> (T K)').tolist()

                # <SOA><stage_1>...codebook 0...<stage_2>...codebook 1-N flattened...<EOA>
                # Or with teacher forcing: <SOA><stage_1>...codebook 0...<stage_2>...all flattened codes...<EOA>
                if not Encoder.codectool.teacher_forcing:
                    codec_ids = ([Encoder.tokenizer.soa, Encoder.tokenizer.stage_1] +
                                 codebook_0_list +
                                 [Encoder.tokenizer.stage_2] +
                                 codebook_rest_list +
                                 [Encoder.tokenizer.eoa])
                else:
                    codec_ids = ([Encoder.tokenizer.soa, Encoder.tokenizer.stage_1] +
                                 codebook_0_list +
                                 [Encoder.tokenizer.stage_2] +
                                 flattened_ids.tolist() + # Use all flattened IDs for teacher forcing
                                 [Encoder.tokenizer.eoa])

                sentence_ids = codec_ids
                doc_ids.extend(sentence_ids)
                sentence_lens.append(len(sentence_ids))

            except Exception as e:
                print(f"Error processing segment in encode_codec_stage_2: {e}")
                print(f"Data ID: {data.get('id', 'unknown')}, Feature Path: {data.get(Encoder.codectool.data_feature, 'unknown')}")
                print(f"Segment Shape: {frames.shape}")
                print(f"FPS: {fps}")


        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
            sentence_lens[-1] += 1

        key = "text" # hardcode key
        ids[key] = doc_ids
        lens[key] = sentence_lens

        return ids, lens, len(json_line) + get_size_in_bytes(raw_codec)


    def encode_token_level_interleave(self, json_line):
        """
        Encodes text and interleaved vocal/instrumental codecs.
        Handles standard interleaving, CoT, and ICL-CoT based on args.
        """
        data = json.loads(json_line)
        ids = {}
        lens = {}

        # --- Initial Data Loading and Validation ---
        required_keys = ['splitted_lyrics', 'vocals_codec', 'instrumental_codec', 'audio_length_in_sec', 'genres', 'id']
        if self.args.use_audio_icl:
            # ICL requires additional keys
            required_keys.extend(['msa', 'codec'])

        if not all(key in data for key in required_keys):
            mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
            print(f"Warning: Missing required keys in data for {data.get('id', 'unknown')} {mode_str}. Skipping.")
            print(f"Missing: {[k for k in required_keys if k not in data]}")
            return {}, {}, len(json_line)
        if not isinstance(data.get('splitted_lyrics'), dict) or 'segmented_lyrics' not in data['splitted_lyrics']:
             mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
             print(f"Warning: Invalid 'splitted_lyrics' format in data for {data.get('id', 'unknown')} {mode_str}. Skipping.")
             return {}, {}, len(json_line)
        if not data['splitted_lyrics']['segmented_lyrics']: # Check if segmented_lyrics is empty
             mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
             print(f"Warning: Empty 'segmented_lyrics' in data for {data.get('id', 'unknown')} {mode_str}. Skipping.")
             return {}, {}, len(json_line)

        segmented_lyrics = data['splitted_lyrics']['segmented_lyrics']

        try:
            raw_codec_vocals = np.load(data['vocals_codec'])
            raw_codec_instrumental = np.load(data['instrumental_codec'])
            # Load mixture codec only if needed for ICL prompt or future use
            raw_codec_mixture = None
            if self.args.use_audio_icl and self.args.audio_prompt_mode == "mixture":
                 raw_codec_mixture = np.load(data['codec'])
        except FileNotFoundError as e:
            mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
            print(f"Error loading codec file {mode_str}: {e}. Skipping data ID {data['id']}.")
            return {}, {}, len(json_line)
        except Exception as e:
            mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
            print(f"Error loading or processing codec npy for {data['id']} {mode_str}: {e}. Skipping.")
            # Estimate size even on error if possible
            bytes_processed = len(json_line)
            try: bytes_processed += get_size_in_bytes(raw_codec_vocals)
            except: pass
            try: bytes_processed += get_size_in_bytes(raw_codec_instrumental)
            except: pass
            try:
                 if raw_codec_mixture is not None: bytes_processed += get_size_in_bytes(raw_codec_mixture)
            except: pass
            return {}, {}, bytes_processed


        # Handle shape mismatch gracefully
        if raw_codec_vocals.shape != raw_codec_instrumental.shape:
            diff = abs(raw_codec_vocals.shape[-1] - raw_codec_instrumental.shape[-1])
            if diff <= 10: # Allow small difference
                min_len = min(raw_codec_vocals.shape[-1], raw_codec_instrumental.shape[-1])
                raw_codec_vocals = raw_codec_vocals[:, :min_len]
                raw_codec_instrumental = raw_codec_instrumental[:, :min_len]
                if DEBUG: print(f"Adjusted codec shapes for {data['id']} due to difference {diff}. New length: {min_len}")
            else:
                mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
                print(f"Warning: Mismatch shape {raw_codec_vocals.shape} vs {raw_codec_instrumental.shape} for {data['id']} {mode_str}. Skipping.")
                bytes_processed = len(json_line) + max(get_size_in_bytes(raw_codec_vocals), get_size_in_bytes(raw_codec_instrumental))
                if raw_codec_mixture is not None: bytes_processed += get_size_in_bytes(raw_codec_mixture)
                return {}, {}, bytes_processed

        # Also check mixture codec shape if loaded
        if raw_codec_mixture is not None and raw_codec_mixture.shape[1] != raw_codec_vocals.shape[1]:
             # Attempt to trim mixture like vocals/instrumental if difference is small
             diff_mix = abs(raw_codec_mixture.shape[-1] - raw_codec_vocals.shape[-1])
             if diff_mix <= 10:
                 raw_codec_mixture = raw_codec_mixture[:, :raw_codec_vocals.shape[1]]
                 if DEBUG: print(f"Adjusted mixture codec shape for {data['id']} to match vocals/instrumental.")
             else:
                 print(f"Warning: Mixture codec shape {raw_codec_mixture.shape} mismatch with vocals/instrumental {raw_codec_vocals.shape} for {data['id']} (ICL-CoT). Skipping.")
                 bytes_processed = len(json_line) + get_size_in_bytes(raw_codec_vocals) + get_size_in_bytes(raw_codec_instrumental) + get_size_in_bytes(raw_codec_mixture)
                 return {}, {}, bytes_processed


        full_length_of_song = data['audio_length_in_sec']
        # Basic checks for validity
        if full_length_of_song <= 0 or raw_codec_vocals.ndim < 2 or raw_codec_vocals.shape[1] == 0:
             mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
             print(f"Warning: Invalid audio length ({full_length_of_song}) or vocal codec shape ({raw_codec_vocals.shape}) for {data['id']} {mode_str}. Skipping.")
             # Calculate bytes processed before returning
             bytes_processed = len(json_line) + get_size_in_bytes(raw_codec_vocals) + get_size_in_bytes(raw_codec_instrumental)
             if raw_codec_mixture is not None: bytes_processed += get_size_in_bytes(raw_codec_mixture)
             return {}, {}, bytes_processed

        fps = raw_codec_vocals.shape[1] / full_length_of_song
        # Relaxed fps check
        if fps > 51 or fps < 49:
            mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
            if DEBUG: print(f"fps={fps} is invalid for {data['id']} {mode_str}, skipping...")
            # Calculate bytes processed before returning
            bytes_processed = len(json_line) + get_size_in_bytes(raw_codec_vocals) + get_size_in_bytes(raw_codec_instrumental)
            if raw_codec_mixture is not None: bytes_processed += get_size_in_bytes(raw_codec_mixture)
            return {}, {}, bytes_processed


        doc_ids = []
        sentence_lens = [] # here sentence means segment
        instruction = self.args.instruction

        # --- Header Construction ---
        if self.args.use_audio_icl:
            # --- Start ICL Prompt Generation ---
            audio_prompt_codec_ids = []
            try:
                # Use the full range covered by lyrics segments for prompt sampling
                prompt_range_start_frame = segmented_lyrics[0].get('codec_frame_start', 0)
                prompt_range_end_frame = segmented_lyrics[-1].get('codec_frame_end', raw_codec_vocals.shape[1])

                # Ensure range is valid
                if prompt_range_start_frame >= prompt_range_end_frame:
                    raise ValueError(f"Invalid prompt range: start={prompt_range_start_frame}, end={prompt_range_end_frame}")

                # Extract relevant segment parts for prompt generation
                raw_codec_vocals_prompt_seg = raw_codec_vocals[:, prompt_range_start_frame:prompt_range_end_frame]
                raw_codec_instrumental_prompt_seg = raw_codec_instrumental[:, prompt_range_start_frame:prompt_range_end_frame]
                raw_codec_mixture_prompt_seg = None
                if raw_codec_mixture is not None:
                    raw_codec_mixture_prompt_seg = raw_codec_mixture[:, prompt_range_start_frame:prompt_range_end_frame]

                vocals_ids_prompt = Encoder.codectool.npy2ids(raw_codec_vocals_prompt_seg)
                instrumental_ids_prompt = Encoder.codectool.npy2ids(raw_codec_instrumental_prompt_seg)

                # Check if ids are valid lists/arrays
                if not isinstance(vocals_ids_prompt, (list, np.ndarray)) or not isinstance(instrumental_ids_prompt, (list, np.ndarray)):
                    raise TypeError("npy2ids did not return list/ndarray for prompt segment")
                if len(vocals_ids_prompt) == 0:
                    raise ValueError("Empty codec IDs generated for prompt segment")

                options_codecs = {}
                codec_step = 1 # How many codec tokens per original frame
                selected_option = self.args.audio_prompt_mode

                if selected_option == "dual":
                    codec_step = 2
                    if len(vocals_ids_prompt) != len(instrumental_ids_prompt):
                        raise ValueError(f"Length mismatch for interleaving prompt: {len(vocals_ids_prompt)} vs {len(instrumental_ids_prompt)}")
                    ids_segment_interleaved = rearrange([np.array(vocals_ids_prompt), np.array(instrumental_ids_prompt)], 'b n -> (n b)')
                    options_codecs['dual'] = ids_segment_interleaved
                elif selected_option == "mixture":
                    if raw_codec_mixture_prompt_seg is None: # Ensure mixture was loaded
                         raise ValueError("Mixture codec selected for prompt but not loaded/available.")
                    mixture_ids_prompt = Encoder.codectool.npy2ids(raw_codec_mixture_prompt_seg)
                    if not isinstance(mixture_ids_prompt, (list, np.ndarray)): raise TypeError("npy2ids failed for mixture prompt")
                    options_codecs['mixture'] = np.array(mixture_ids_prompt)
                elif selected_option == "inst":
                    options_codecs['inst'] = np.array(instrumental_ids_prompt)
                elif selected_option == "vocal":
                    options_codecs['vocal'] = np.array(vocals_ids_prompt)
                else:
                    raise ValueError(f"Invalid audio_prompt_mode: {selected_option}")

                # Determine prompt length in codec frames/tokens
                audio_prompt_length_in_secs = inverse_transform_sampling(Encoder.cdf_values, Encoder.x_values).item()
                audio_prompt_length_in_frames = int(audio_prompt_length_in_secs * fps)
                audio_prompt_length_in_codec_tokens = audio_prompt_length_in_frames * codec_step

                segment_duration_frames = prompt_range_end_frame - prompt_range_start_frame
                segment_duration_codec_tokens = segment_duration_frames * codec_step

                # Ensure prompt length is valid and fits within the segment
                if audio_prompt_length_in_codec_tokens <= 0:
                    audio_prompt_length_in_codec_tokens = int(1 * fps * codec_step) # Default to 1 second
                if audio_prompt_length_in_codec_tokens >= segment_duration_codec_tokens:
                    audio_prompt_length_in_codec_tokens = segment_duration_codec_tokens // 2 # Take half if too long
                    if DEBUG: print(f"Prompt length adjusted to {audio_prompt_length_in_codec_tokens} tokens (half segment) for {data['id']}")

                # --- Sample start position for the prompt ---
                max_start_token_index = segment_duration_codec_tokens - audio_prompt_length_in_codec_tokens
                if max_start_token_index < 0 : max_start_token_index = 0

                prompt_start_token_idx = 0
                # Try sampling from chorus if available
                chorus_list = [s for s in data.get('msa', []) if s.get('label') == 'chorus']
                if chorus_list:
                    random_chorus = random.choice(chorus_list)
                    chorus_start_sec = random_chorus.get('start', 0)
                    chorus_end_sec = random_chorus.get('end', full_length_of_song)

                    # Convert chorus times relative to the start of the lyrics segment range
                    chorus_start_frame_relative = max(0, int(chorus_start_sec * fps) - prompt_range_start_frame)
                    chorus_end_frame_relative = min(segment_duration_frames, int(chorus_end_sec * fps) - prompt_range_start_frame)

                    chorus_start_token_relative = chorus_start_frame_relative * codec_step
                    chorus_end_token_relative = chorus_end_frame_relative * codec_step

                    # Define valid start range within the chorus
                    chorus_max_start_token = chorus_end_token_relative - audio_prompt_length_in_codec_tokens
                    chorus_min_start_token = chorus_start_token_relative

                    if chorus_max_start_token > chorus_min_start_token:
                        prompt_start_token_idx = random.randint(chorus_min_start_token, chorus_max_start_token)
                    else:
                        prompt_start_token_idx = random.randint(0, max_start_token_index) # Fallback
                else:
                    prompt_start_token_idx = random.randint(0, max_start_token_index) # Random start

                prompt_end_token_idx = prompt_start_token_idx + audio_prompt_length_in_codec_tokens
                audio_prompt_codec_array = options_codecs[selected_option][prompt_start_token_idx:prompt_end_token_idx]

                # Optional: Filter prompts with low variation
                retry_count=0
                min_unique_ratio = 0.1
                while (len(np.unique(audio_prompt_codec_array)) < len(audio_prompt_codec_array) * min_unique_ratio) and retry_count < 5:
                    if DEBUG: print(f"Retrying prompt sampling due to low variation ({len(np.unique(audio_prompt_codec_array))} unique) for {data['id']}")
                    prompt_start_token_idx = random.randint(0, max_start_token_index)
                    prompt_end_token_idx = prompt_start_token_idx + audio_prompt_length_in_codec_tokens
                    audio_prompt_codec_array = options_codecs[selected_option][prompt_start_token_idx:prompt_end_token_idx]
                    retry_count += 1

                if retry_count == 5:
                    print(f"Warning: Could not find suitable audio prompt with enough variation for {data['id']} after 5 retries.")

                audio_prompt_codec_ids = ([Encoder.tokenizer.soa] + Encoder.codectool.sep_ids +
                                        list(audio_prompt_codec_array) +
                                        [Encoder.tokenizer.eoa])

            except Exception as e:
                print(f"Error generating ICL audio prompt for {data['id']}: {e}")
                print("Skipping sample due to ICL prompt error.")
                # Calculate bytes processed before returning
                bytes_processed = len(json_line) + get_size_in_bytes(raw_codec_vocals) + get_size_in_bytes(raw_codec_instrumental)
                if raw_codec_mixture is not None: bytes_processed += get_size_in_bytes(raw_codec_mixture)
                return {}, {}, bytes_processed # Skip sample


            # Construct ICL-CoT Header
            genre_str = '[Genre] ' + data['genres']
            complete_lyrics = '\n'.join([l.get('line_content', '') for l in segmented_lyrics])
            # Format: <Instruction> \n <Genre> \n <Lyrics> [start_of_reference] <Prompt> [end_of_reference]
            head = f'{instruction}\n{genre_str}\n{complete_lyrics}'
            head_ids = (Encoder.tokenizer.tokenize(head) +
                        Encoder.tokenizer.tokenize("[start_of_reference]") +
                        audio_prompt_codec_ids +
                        Encoder.tokenizer.tokenize("[end_of_reference]"))
            doc_ids.extend(head_ids)
            sentence_lens.append(len(head_ids))
            # --- End ICL Header ---

        elif self.args.cot:
            # Construct standard CoT Header (no audio prompt)
            genre_str = '[Genre] ' + data['genres']
            complete_lyrics = '\n'.join([l.get('line_content', '') for l in segmented_lyrics])
            # Format: <Instruction> \n <Genre> \n <Lyrics>
            head = f'{instruction}\n{genre_str}\n{complete_lyrics}'
            head_ids = Encoder.tokenizer.tokenize(head)
            doc_ids.extend(head_ids)
            sentence_lens.append(len(head_ids))
        # Else: No CoT, no ICL - header is implicitly handled per segment (instruction prepended)


        # --- Process Individual Segments ---
        for segment in segmented_lyrics:
            duration = segment.get('duration')
            frame_start = segment.get('codec_frame_start')
            frame_end = segment.get('codec_frame_end')
            line_content = segment.get('line_content')

            # Basic validation of segment data
            if duration is None or frame_start is None or frame_end is None or line_content is None:
                if DEBUG: print(f"Skipping segment due to missing keys: {segment} in {data['id']}")
                continue
            # Frame indices validity already checked for the whole song's fps calculation
            if not (0 <= frame_start < frame_end <= raw_codec_vocals.shape[1]):
                 if DEBUG: print(f"Invalid frame indices for segment in {data['id']}: start={frame_start}, end={frame_end}, total={raw_codec_vocals.shape[1]}. Skipping.")
                 continue
            if frame_end - frame_start <= 0:
                 if DEBUG: print(f"Segment frame length is zero or negative in {data['id']}: {frame_end - frame_start}. Skipping.")
                 continue
            # Minimum duration check (e.g., > 1 sec for target, or based on fps)
            min_target_segment_duration_sec = 1.0
            if self.args.use_audio_icl and duration < min_target_segment_duration_sec:
                 if DEBUG: print(f"Skipping target segment in {data['id']} (ICL) because duration {duration} < {min_target_segment_duration_sec}s")
                 continue
            # Check based on fps if not ICL (ensure at least 1 second)
            elif not self.args.use_audio_icl and frame_end - frame_start < fps:
                if DEBUG: print(f"Segment frame too short in {data['id']}: length={frame_end - frame_start} (< {fps}), skipping...")
                continue


            raw_codec_vocals_segment = raw_codec_vocals[:, frame_start:frame_end]
            raw_codec_instrumental_segment = raw_codec_instrumental[:, frame_start:frame_end]

            # --- Tokenize Text ---
            text_ids = []
            text = "" # Initialize text for potential error printing
            if self.args.cot or self.args.use_audio_icl: # CoT/ICL uses only line content for segment text
                text = line_content
            else: # Standard non-CoT mode
                text = instruction + '\n' + line_content
                 # Apply instruction dropout if enabled and not CoT/ICL
                if self.args.instruction_dropout_rate > 0.0 and np.random.rand() < self.args.instruction_dropout_rate:
                    text = line_content
            text_ids = Encoder.tokenizer.tokenize(text)


            # --- Process Codec ---
            try:
                vocals_ids_seg = Encoder.codectool.npy2ids(raw_codec_vocals_segment)
                instrumental_ids_seg = Encoder.codectool.npy2ids(raw_codec_instrumental_segment)

                if not isinstance(vocals_ids_seg, (list, np.ndarray)) or not isinstance(instrumental_ids_seg, (list, np.ndarray)):
                    raise TypeError("npy2ids did not return a list or ndarray for segment")
                if len(vocals_ids_seg) != len(instrumental_ids_seg):
                     mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
                     print(f"Warning: Mismatch target vocal/inst IDs ({len(vocals_ids_seg)}/{len(instrumental_ids_seg)}) for {data['id']} {mode_str}. Skipping segment.")
                     continue
                if len(vocals_ids_seg) == 0: # Skip empty segments
                    if DEBUG: print(f"Skipping segment in {data['id']} because resulting codec IDs are empty.")
                    continue

                ids_segment_interleaved = rearrange([np.array(vocals_ids_seg), np.array(instrumental_ids_seg)], 'b n -> (n b)')
                ids_segment_interleaved_list = list(ids_segment_interleaved)

                # --- Construct Segment Tokens ---
                segment_tokens = []
                if self.args.cot or self.args.use_audio_icl:
                    # Format for CoT/ICL-CoT: [start_of_segment] <text> <SOA> <sep> <interleaved_codec> <EOA> [end_of_segment]
                    segment_tokens = (Encoder.tokenizer.tokenize('[start_of_segment]') +
                                     text_ids +
                                     [Encoder.tokenizer.soa] + Encoder.codectool.sep_ids +
                                     ids_segment_interleaved_list +
                                     [Encoder.tokenizer.eoa] +
                                     Encoder.tokenizer.tokenize('[end_of_segment]'))
                else:
                    # Standard non-CoT format: <text> <SOA> <sep> <interleaved_codec> <EOA>
                    codec_tokens = ([Encoder.tokenizer.soa] + Encoder.codectool.sep_ids +
                                    ids_segment_interleaved_list +
                                    [Encoder.tokenizer.eoa])
                    segment_tokens = text_ids + codec_tokens

                doc_ids.extend(segment_tokens)
                sentence_lens.append(len(segment_tokens))

            except Exception as e:
                mode_str = "(ICL-CoT)" if self.args.use_audio_icl else ""
                print(f"Error processing segment in encode_token_level_interleave {mode_str}: {e}")
                print(f"Data ID: {data['id']}")
                print(f"Segment: {segment}")
                print(f"Text Input: {text}") # Print the text that was tokenized


        # --- Finalize Document ---
        if len(doc_ids) > 0 and self.args.append_eod:
            # Add EOD only if we have successfully processed something (header or segments)
            if sentence_lens:
                 doc_ids.append(Encoder.tokenizer.eod)
                 sentence_lens[-1] += 1
            else:
                 if DEBUG: print(f"Skipping EOD for {data['id']} as no valid segments/header were processed.")


        key = "text" # hardcode key
        ids[key] = doc_ids
        lens[key] = sentence_lens

        bytes_processed = len(json_line) + get_size_in_bytes(raw_codec_vocals) + get_size_in_bytes(raw_codec_instrumental)
        if raw_codec_mixture is not None: # Add mixture size if it was loaded
             bytes_processed += get_size_in_bytes(raw_codec_mixture)
        return ids, lens, bytes_processed



class Partition(object):
    """Handles partitioning and processing of input files."""
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        """Prints processing statistics."""
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            if elapsed > 0:
                 docs_per_sec = count / elapsed
                 mbs = total_bytes_processed / elapsed / 1024 / 1024
                 print(f"Processed {count} documents",
                       f"({docs_per_sec:.2f} docs/s, {mbs:.2f} MB/s).",
                       file=sys.stderr)
            else:
                 print(f"Processed {count} documents (elapsed time is zero).", file=sys.stderr)


    def split_sentences(self, file_name):
        """Splits documents into sentences (if enabled)."""
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name, "for sentence splitting")
        try:
            fin = open(input_file_name, 'r', encoding='utf-8')
            fout = open(output_file_name, 'w', encoding='utf-8') # Ensure utf-8 for output
        except Exception as e:
            print(f"Error opening files for sentence splitting: {e}")
            return


        encoder = Encoder(self.args)
        # Setup multiprocessing pool
        try:
             pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
             # Assuming encoder.split exists and handles sentence splitting based on args.lang
             split_docs = pool.imap(encoder.split, fin, 32) # Use imap for memory efficiency
        except Exception as e:
             print(f"Error creating multiprocessing pool or starting imap: {e}")
             fin.close()
             fout.close()
             return

        proc_start = time.time()
        total_bytes_processed = 0
        processed_count = 0
        try:
            for i, result in enumerate(split_docs, start=1):
                # Assuming encoder.split returns (doc_string, bytes_processed)
                if isinstance(result, tuple) and len(result) == 2:
                     doc, bytes_processed = result
                     total_bytes_processed += bytes_processed
                     fout.write(doc + "\n") # Fixed newline
                     processed_count = i
                     self.print_processing_stats(i, proc_start, total_bytes_processed)
                else:
                     print(f"Warning: Unexpected result format from encoder.split: {result}")
        except Exception as e:
            print(f"Error during sentence splitting processing: {e}")
        finally:
            pool.close()
            pool.join()
            fin.close()
            fout.close()
            print(f"Finished sentence splitting for {input_file_name}. Processed {processed_count} documents.")


    def process_json_file(self, file_name):
        """Processes a JSONL file, encoding documents based on arguments."""
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name, "for processing")
        try:
             # Handle potential gzipped files
             if input_file_name.endswith(".gz"):
                 fin = gzip.open(input_file_name, 'rt', encoding='utf-8')
             else:
                 fin = open(input_file_name, 'r', encoding='utf-8')
        except Exception as e:
             print(f"Error opening input file {input_file_name}: {e}")
             return


        startup_start = time.time()
        encoder = Encoder(self.args)
        try:
            tokenizer = _MMSentencePieceTokenizer(self.args.tokenizer_model, vocab_extra_ids=self.args.vocab_extra_ids)
            # Initialize encoder (loads tokenizer, codectool etc.)
            encoder.initializer()
            # Pass tokenizer explicitly if not done in initializer
            Encoder.tokenizer = tokenizer
        except Exception as e:
            print(f"Error initializing tokenizer or encoder: {e}")
            fin.close()
            return


        # Determine encoding function based on args
        encode_func = None
        if self.args.order == "stage2":
            print("Using encode_codec_stage_2")
            encode_func = encoder.encode_codec_stage_2
        # MERGED: Handle both ICL and non-ICL token interleaving
        elif self.args.use_token_level_interleave or (self.args.use_audio_icl and self.args.cot):
            if self.args.use_audio_icl:
                print("Using encode_token_level_interleave (ICL-CoT mode)")
            else:
                print("Using encode_token_level_interleave (standard/CoT mode)")
            encode_func = encoder.encode_token_level_interleave
        elif self.args.order in ["textfirst", "audiofirst"]:
             print(f"Using encode_mix_text_and_codec (order: {self.args.order})")
             encode_func = encoder.encode_mix_text_and_codec
        else:
             print(f"Error: Could not determine appropriate encoder function based on args: order={self.args.order}, use_audio_icl={self.args.use_audio_icl}, cot={self.args.cot}, use_token_level_interleave={self.args.use_token_level_interleave}")
             fin.close()
             return # Exit if no valid encoder function determined


        # Setup multiprocessing pool or run in debug mode
        encoded_docs = []
        pool = None
        if not DEBUG:
            try:
                # Pass necessary class variables if initializer doesn't handle them correctly
                pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
                encoded_docs = pool.imap(encode_func, fin, 32)
            except Exception as e:
                 print(f"Error creating multiprocessing pool or starting imap: {e}")
                 fin.close()
                 if pool: pool.close(); pool.join()
                 return
        else: # DEBUG mode
            print("Running in DEBUG mode (single process)")
            # Prepare list for debug processing
            debug_docs_list = []
            for line in fin:
                 try:
                     result = encode_func(line)
                     # Check if result is valid before appending
                     if isinstance(result, tuple) and len(result) == 3:
                          ids, lens, b_processed = result
                          if ids and lens: # Ensure ids and lens are not empty
                              debug_docs_list.append(result)
                          # else: Print debug message if needed
                     # else: Print warning about invalid result format
                 except Exception as e:
                      print(f"Error processing line in DEBUG mode: {e}")
            encoded_docs = iter(debug_docs_list) # Make it iterable like imap


        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        # Initialize builders for specified keys (usually just 'text')
        try:
            # Use tokenizer.vocab_size which should be available after initialization
            dtype = indexed_dataset.DType.optimal_dtype(Encoder.tokenizer.vocab_size)
            for key in self.args.json_keys:
                # Skip 'codec' if 'text' is present, assuming merged processing
                if key == 'codec' and "text" in self.args.json_keys:
                    print("[Info] 'codec' key specified but will be processed as part of 'text'.")
                    continue
                output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
                output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"
                builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(output_bin_files[key], dtype=dtype)
        except Exception as e:
            print(f"Error initializing IndexedDataset builders: {e}")
            fin.close()
            if pool: pool.close(); pool.join()
            return


        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        processed_count = 0
        print(f"Time to startup processing for {input_file_name}: {startup_end - startup_start:.2f} seconds")

        try:
            for i, result in enumerate(encoded_docs, start=1):
                 # Validate result format from encode function
                 if not (isinstance(result, tuple) and len(result) == 3):
                     print(f"Warning: Invalid result format received from encoder: {type(result)}. Skipping item {i}.")
                     continue

                 doc, sentence_lens_map, bytes_processed = result
                 total_bytes_processed += bytes_processed

                 # Check if doc is empty (might happen if a sample is skipped by the encoder)
                 if not doc:
                      if DEBUG: print(f"Skipping empty document result at index {i}")
                      continue

                 for key in doc.keys():
                     if key in builders:
                          # Ensure doc[key] and sentence_lens_map[key] are valid
                          if not isinstance(doc[key], list) or not isinstance(sentence_lens_map.get(key), list):
                               print(f"Warning: Invalid data format for key '{key}' in doc/lens map at index {i}. Skipping.")
                               continue
                          # Add document if lens are provided and match structure (simple list of lengths)
                          doc_lens = sentence_lens_map.get(key)
                          if doc_lens is not None:
                               builders[key].add_document(doc[key], doc_lens)
                               processed_count = i # Update count only on successful add
                          else:
                               print(f"Warning: Missing sentence lengths for key '{key}' at index {i}. Skipping.")
                     # else: Silently ignore keys not specified in --json-keys

                 self.print_processing_stats(processed_count, proc_start, total_bytes_processed)

        except Exception as e:
            print(f"Error during document processing loop: {e}")
        finally:
            # Ensure pool is closed if it exists
            if pool:
                pool.close()
                pool.join()
            # Finalize builders
            finalized_keys = []
            for key in builders.keys():
                try:
                    print(f"Finalizing index for key '{key}'...")
                    builders[key].finalize(output_idx_files[key])
                    finalized_keys.append(key)
                except Exception as e:
                    print(f"Error finalizing builder for key '{key}': {e}")

            fin.close()
            print(f"Finished processing {input_file_name}. Processed {processed_count} documents.")
            if finalized_keys:
                 print(f"Finalized outputs for keys: {finalized_keys}")
            else:
                 print("Warning: No output builders were finalized.")


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON file(s) (glob pattern supported, e.g., "data/*.jsonl")')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space-separated list of keys to extract from json (usually just "text" for combined processing).')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences (requires NLTK).')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting (currently not implemented in EncoderBase.split).')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True, default='MMSentencePieceTokenizer',
                       choices=['MMSentencePieceTokenizer'], # Restricted choices based on usage
                       help='Tokenizer type (currently only MMSentencePieceTokenizer supported).')
    group.add_argument('--tokenizer-model', type=str, required=True,
                       help='Path to the SentencePiece tokenizer model.')
    # Removed vocab_file, merge_file as they are typically not used with SentencePiece directly in this context
    # group.add_argument('--vocab-size', type=int, default=786, help='Size of vocab (legacy, may not be needed).')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of extra IDs in the vocabulary.')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of each document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language for NLTK sentence splitting (if --split-sentences is used).')

    group = parser.add_argument_group(title='codec and processing mode')
    group.add_argument('--codec-type', type=str, required=True,
                       choices=['dac16k', 'dac44k', 'xcodec', 'mert', 'hubert', 'semantic/s', 'semantic/a', 'semanticodec'],
                       help="Type of codec used to generate '.npy' files.")
    group.add_argument('--instruction', type=str, default="Generate audio from the given text condition.",
                       help='Instruction text prepended in some modes.')
    group.add_argument('--instruction-dropout-rate', type=float, default=0.0,
                       help='Dropout rate for the instruction text (if applicable).')
    group.add_argument('--order', type=str, required=True,
                       choices=['textfirst', 'audiofirst', # Original modes
                                'stage2',                 # Stage 2 codec processing
                               # Add other potential future modes here if needed
                               # 'text_icl_audio', 'icl_text_audio'
                               ],
                       help='Processing order and mode selection.')
    group.add_argument('--use-token-level-interleave', action='store_true',
                       help='Enable token-level interleaving of vocal/instrumental codecs.')
    group.add_argument('--cot', action='store_true',
                       help='Use Chain-of-Thought formatting (requires specific data structure).')
    group.add_argument('--use-audio-icl', action='store_true',
                       help='Enable In-Context Learning with an audio prompt.')
    group.add_argument('--audio-prompt-mode', type=str, default="dual",
                       choices=['mixture', 'dual', 'inst', 'vocal'],
                       help='Source for the audio prompt in ICL mode.')
    group.add_argument('--audio-prompt-len', type=int, default=30, help='Length of audio prompt (now sampled) around 30s.')
    group.add_argument('--min-icl-song-duration-sec', type=float, default=40.0,
                        help='Minimum song duration in seconds required to attempt ICL processing.')


    group = parser.add_argument_group(title='stage 2 specific')
    group.add_argument('--quantizer-begin', type=int, default=0, # Default to 0 if not specified
                       help='Index of the first quantizer layer to use for stage 2.')
    group.add_argument('--n-quantizer', type=int, default=8, # Default to 8 if not specified
                       help='Number of quantizer layers to use for stage 2.')
    group.add_argument('--teacher-forcing', action='store_true',
                       help='Use teacher forcing for stage 2 (target includes all flattened codes).')
    group.add_argument('--data-feature', type=str, default='codec',
                       help='JSON key pointing to the codec .npy file for stage 2.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path prefix for binary output files (e.g., "output/processed_data").')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes.')
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions to process in parallel (requires input glob pattern).')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval for logging processing progress.')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Preserve original order when using partitions > 1 (slower).')

    args = parser.parse_args()
    args.keep_empty = False # Keep this? Seems related to older dataset versions

    # Add derived arguments or validation
    if args.use_audio_icl and not args.cot:
         parser.error("--use-audio-icl currently requires --cot.")
    if args.cot and not args.use_token_level_interleave and not args.use_audio_icl:
         print("Warning: --cot is enabled but neither --use-token-level-interleave nor --use-audio-icl is set. Ensure your encoder handles this.")

    # Dummy args for compatibility if EncoderBase or other parts expect them
    args.rank = 0 # Usually set by distributed environment
    args.make_vocab_size_divisible_by = 128 # Often needed for model parallelism efficiency
    args.tensor_model_parallel_size = 1 # Usually set by distributed environment

    return args

def main():
    """Main function to orchestrate data partitioning and processing."""
    args = get_args()
    print("Arguments received:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    # Setup NLTK path if splitting sentences
    if args.split_sentences:
        if nltk_available:
             # Ensure NLTK data path is configured (e.g., via environment variable NLTK_DATA)
             nltk_data_path = os.environ.get("NLTK_DATA")
             if nltk_data_path:
                 print(f"Using NLTK data path: {nltk_data_path}")
                 # Check if 'punkt' is available, download if not (be cautious with auto-downloads)
                 try:
                      nltk.data.find('tokenizers/punkt')
                 except nltk.downloader.DownloadError:
                      print("NLTK 'punkt' tokenizer not found. Attempting download...")
                      try:
                           nltk.download("punkt", quiet=False, download_dir=nltk_data_path)
                      except Exception as e:
                           raise RuntimeError(f"Failed to download NLTK 'punkt' tokenizer. Please install it manually or check NLTK_DATA path. Error: {e}")
             else:
                  print("Warning: NLTK_DATA environment variable not set. NLTK will use default paths.")
                  try:
                       nltk.data.find('tokenizers/punkt')
                  except Exception:
                       raise RuntimeError("NLTK 'punkt' tokenizer not found. Please install it or set NLTK_DATA.")

        else:
            raise ImportError("nltk library required for --split-sentences is not available.")


    # --- File Handling and Partitioning ---
    in_ss_out_names = [] # List to store dictionaries of file names for each partition
    input_pattern = args.input
    output_prefix = args.output_prefix
    num_partitions = args.partitions

    if num_partitions == 1:
        # Check if input is a single file or a pattern that resolves to one file
        in_file_names = glob.glob(input_pattern)
        if not in_file_names:
            print(f"Error: No input files found matching pattern: {input_pattern}")
            sys.exit(1)
        if len(in_file_names) > 1:
            print(f"Warning: Input pattern '{input_pattern}' matched multiple files, but partitions=1. Using first file: {in_file_names[0]}")
        input_file = in_file_names[0]
        file_name_base, extension = os.path.splitext(os.path.basename(input_file))
        # Use output_prefix directly for the single partition's output files
        partition_output_prefix = output_prefix
        sentence_split_file = f"{partition_output_prefix}_ss{extension}" # Suffix added to output prefix

        file_names = {
            'partition': input_file, # Original input file
            'sentence_split': sentence_split_file, # Temporary file if splitting sentences
            'output_prefix': partition_output_prefix # Prefix for final .bin/.idx files
        }
        in_ss_out_names.append(file_names)
    else:
        # Handle multiple partitions
        in_file_names = glob.glob(input_pattern)
        if not in_file_names:
            print(f"Error: No input files found matching pattern: {input_pattern} for {num_partitions} partitions.")
            sys.exit(1)
        print(f"Found {len(in_file_names)} input files matching pattern.")

        # Create partition file names (temporary files)
        partition_base = os.path.join(os.path.dirname(output_prefix), "partition_files")
        os.makedirs(partition_base, exist_ok=True) # Ensure directory exists

        for idx in range(num_partitions):
             # Generate names for temporary partition files and their outputs
             # Use a consistent naming scheme based on the overall output prefix and partition index
             partition_file_path = os.path.join(partition_base, f"input_part_{idx:03d}.jsonl")
             partition_output_prefix = f"{output_prefix}_part_{idx:03d}" # Partition-specific output prefix
             sentence_split_file = f"{partition_output_prefix}_ss.jsonl" # Temp sentence split file for this partition

             in_ss_out_name = {
                 'partition': partition_file_path, # Path to temp input partition file
                 'sentence_split': sentence_split_file, # Path to temp sentence split file
                 'output_prefix': partition_output_prefix # Prefix for this partition's .bin/.idx
                 }
             in_ss_out_names.append(in_ss_out_name)

        # Check if temporary partition files already exist
        partitions_present = check_files_exist(in_ss_out_names, 'partition', num_partitions)
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', num_partitions) if args.split_sentences else True # Assume present if not splitting

        if not partitions_present:
            print("Creating temporary partition files...")
            # Distribute lines from input files into partition files
            partitioned_input_files = [open(name['partition'], 'w', encoding='utf-8') for name in in_ss_out_names]

            line_count = 0
            processed_files_count = 0
            try:
                for in_file_name in sorted(in_file_names): # Sort for determinism if needed
                    print(f"Reading input file: {in_file_name}")
                    # Handle gzip
                    if in_file_name.endswith(".gz"):
                        fin = gzip.open(in_file_name, 'rt', encoding='utf-8')
                    else:
                        fin = open(in_file_name, 'r', encoding='utf-8')

                    with fin:
                         for line in fin:
                              # Distribute lines round-robin or sequentially based on args.keep_sequential_samples
                              # Simple round-robin distribution:
                              target_partition_index = line_count % num_partitions
                              partitioned_input_files[target_partition_index].write(line)
                              line_count += 1
                    processed_files_count += 1
                    print(f"Finished reading {in_file_name}. Total lines distributed so far: {line_count}")

            except Exception as e:
                 print(f"Error distributing lines to partitions: {e}")
                 # Clean up open files before exiting
                 for f in partitioned_input_files: f.close()
                 sys.exit(1)
            finally:
                for f in partitioned_input_files: f.close()
            print(f"Finished creating {num_partitions} partition files. Total lines processed: {line_count} from {processed_files_count} input files.")
        else:
            print("Temporary partition files already exist. Skipping creation.")


    # --- Worker Setup ---
    if args.workers <= 0:
        print("Error: Number of workers must be positive.")
        sys.exit(1)
    if args.workers % num_partitions != 0:
        print(f"Warning: Number of workers ({args.workers}) is not divisible by the number of partitions ({num_partitions}). This might lead to uneven load.")
        workers_per_partition = args.workers // num_partitions
        if workers_per_partition == 0: workers_per_partition = 1 # Ensure at least one worker per partition
        print(f"Assigning approximately {workers_per_partition} workers per partition.")
    else:
        workers_per_partition = args.workers // num_partitions
        print(f"Assigning {workers_per_partition} workers per partition.")

    partition_handler = Partition(args, workers_per_partition)


    # --- Optional: Sentence Splitting of Partition Files ---
    input_key_for_processing = 'partition' # Default: process the initial partition files
    if args.split_sentences:
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', num_partitions)
        if not split_sentences_present:
            print("Splitting sentences in partition files...")
            processes = []
            for name in in_ss_out_names:
                try:
                    p = multiprocessing.Process(target=partition_handler.split_sentences,
                                                args=((name['partition'], name['sentence_split']),))
                    p.start()
                    processes.append(p)
                except Exception as e:
                    print(f"Error starting sentence splitting process for {name['partition']}: {e}")
                    # Handle error: maybe terminate already started processes?

            # Wait for sentence splitting processes to complete
            for i, p in enumerate(processes):
                try:
                    p.join()
                    if p.exitcode != 0:
                         print(f"Warning: Sentence splitting process for partition {i} exited with code {p.exitcode}")
                except Exception as e:
                    print(f"Error joining sentence splitting process for partition {i}: {e}")
            print("Sentence splitting complete.")
            input_key_for_processing = 'sentence_split' # Process the sentence-split files next
        else:
             print("Sentence-split files already exist. Skipping splitting.")
             input_key_for_processing = 'sentence_split'


    # --- Main Encoding Process ---
    print(f"Starting encoding process using input key: '{input_key_for_processing}'...")
    processes = []
    for name in in_ss_out_names:
        try:
            p = multiprocessing.Process(target=partition_handler.process_json_file,
                                        args=((name[input_key_for_processing], name['output_prefix']),))
            p.start()
            processes.append(p)
        except Exception as e:
            print(f"Error starting encoding process for {name[input_key_for_processing]}: {e}")


    # Wait for encoding processes to complete
    for i, p in enumerate(processes):
        try:
            p.join()
            if p.exitcode != 0:
                 print(f"Warning: Encoding process for partition {i} exited with code {p.exitcode}")
        except Exception as e:
            print(f"Error joining encoding process for partition {i}: {e}")

    print("All encoding processes finished.")

    if num_partitions == 1:
        print("Processing complete for single partition.")
        return # Nothing more to do

    # --- Merge Bin/Idx Partitions ---
    print("Merging partition results...")
    level = "sentence" if args.split_sentences else "document"
    final_output_bin_files = {}
    final_output_idx_files = {}
    final_builders = {}

    try:
        # Re-initialize tokenizer to get vocab size for final builders
        tokenizer = _MMSentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
        dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    except Exception as e:
        print(f"Error re-initializing tokenizer for merging: {e}")
        sys.exit(1)


    for key in args.json_keys:
        # Skip 'codec' if 'text' is present (as it was handled during processing)
        if key == 'codec' and "text" in args.json_keys:
            continue

        print(f"Merging results for key: '{key}'")
        final_output_bin_files[key] = f"{args.output_prefix}_{key}_{level}.bin"
        final_output_idx_files[key] = f"{args.output_prefix}_{key}_{level}.idx"

        try:
            final_builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
                final_output_bin_files[key], dtype=dtype
            )

            # Add indices from each partition's output
            num_indices_added = 0
            for name in in_ss_out_names:
                partition_output_prefix = name['output_prefix'] # e.g., "output/data_part_000"
                partition_index_prefix = f"{partition_output_prefix}_{key}_{level}" # e.g., "output/data_part_000_text_document"

                # Check if the partition's bin and idx files exist before adding
                partition_bin_file = f"{partition_index_prefix}.bin"
                partition_idx_file = f"{partition_index_prefix}.idx"
                if os.path.exists(partition_bin_file) and os.path.exists(partition_idx_file):
                    try:
                         final_builders[key].add_index(partition_index_prefix)
                         num_indices_added += 1
                         if DEBUG: print(f"Added index: {partition_index_prefix}")
                    except Exception as e:
                         print(f"Error adding index {partition_index_prefix} for key '{key}': {e}")
                else:
                    print(f"Warning: Index files not found for partition {partition_index_prefix}. Skipping.")


            # Finalize the merged index
            if num_indices_added > 0:
                 print(f"Finalizing merged index for key '{key}' with {num_indices_added} partitions...")
                 final_builders[key].finalize(final_output_idx_files[key])
                 print(f"Finalized merged index: {final_output_idx_files[key]}")
            else:
                 print(f"Warning: No partition indices were added for key '{key}'. Cannot finalize merged index.")
                 # Clean up potentially created empty bin file
                 if os.path.exists(final_output_bin_files[key]):
                      try: os.remove(final_output_bin_files[key])
                      except OSError: pass


        except Exception as e:
            print(f"Error merging or finalizing index for key '{key}': {e}")

    print("Finished merging partitions.")
    # Optional: Clean up temporary partition files here if desired


if __name__ == '__main__':
    main() 