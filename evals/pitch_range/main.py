import numpy as np
import librosa
import time
import argparse
import torch
from extract_pitch_values_from_audio.src import RMVPE
import os
from pathlib import Path
from tqdm import tqdm

def process_audio(rmvpe, audio_path, output_path, device, hop_length, threshold):
    """Process an audio file in 10-second chunks and save the results."""
    # Load the audio file
    audio, sr = librosa.load(str(audio_path), sr=None)
    chunk_size = 10 * sr
    # pad to make the audio length to be multiple of hop_length
    audio = np.pad(audio, (0, chunk_size - len(audio) % chunk_size), mode='constant')
    
    # Calculate chunk size in samples (10 seconds * sample rate)
    total_chunks = int(np.round(len(audio) / chunk_size))
    
    # Initialize arrays to store results
    all_f0 = []
    total_infer_time = 0
    
    # Process each chunk
    for i in tqdm(range(total_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # Process the chunk
        t = time.time()
        f0_chunk = rmvpe.infer_from_audio(chunk, sr, device=device, thred=threshold, use_viterbi=True)
        chunk_infer_time = time.time() - t
        total_infer_time += chunk_infer_time
        
        # Append results
        all_f0.extend(f0_chunk)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # remove all 0 in the f0
    all_f0 = np.array(all_f0)
    all_f0 = all_f0[all_f0 != 0]

    # convert all_f0 to a list
    all_f0 = all_f0.tolist()
    
    # Save the results
    with open(output_path, 'w') as f:
        for f0 in all_f0:
            f.write(f'{f0:.2f}\n')
    
    return total_infer_time, len(audio) / sr  # Return total inference time and audio duration

def main():
    input_dir = Path("/root/yue_pitch_evals/yue_vs_others_sep")
    output_dir = Path("/root/yue_pitch_evals/yue_vs_others_sep_pitch")
    device = "cuda"
    
    print(f'Using device: {device}')
    print('Loading model...')
    rmvpe = RMVPE("model.pt", hop_length=160)
    
    # Find all WAV files in input directory and subdirectories
    wav_files = list(input_dir.rglob('*.Vocals.mp3'))
    print(f'Found {len(wav_files)} WAV files to process')
    
    total_time = 0
    total_audio_duration = 0
    
    # Process each WAV file
    for wav_path in tqdm(wav_files, desc="Processing files"):
        # Calculate relative path to maintain directory structure
        rel_path = wav_path.relative_to(input_dir)
        # Create output path with .txt extension
        output_path = output_dir / str(rel_path).replace('.Vocals.mp3', '.txt')
        
        try:
            infer_time, audio_duration = process_audio(
                rmvpe, wav_path, output_path, device, 
                160, 0.03
            )
            total_time += infer_time
            total_audio_duration += audio_duration
            
            tqdm.write(f'Processed {wav_path.name}')
            tqdm.write(f'Time: {infer_time:.2f}s, RTF: {infer_time/audio_duration:.2f}')
            
        except Exception as e:
            tqdm.write(f'Error processing {wav_path}: {str(e)}')
            continue
    
    print('\nProcessing complete!')
    print(f'Total processing time: {total_time:.2f}s')
    print(f'Average RTF: {total_time/total_audio_duration:.2f}')

if __name__ == '__main__':
    main()