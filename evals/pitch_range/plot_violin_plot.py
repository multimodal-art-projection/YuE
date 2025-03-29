import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import math

def freq_to_midi_note(frequency):
    if frequency <= 0:
        return None
    return 69 + 12 * math.log2(frequency / 440.0)

def get_persistent_notes(frequencies, persistence_frames=4):
    if len(frequencies) < persistence_frames:
        return []
    
    persistent_notes = []
    current_note = None
    persistence_count = 0
    
    for freq in frequencies:
        if freq <= 0:
            current_note = None
            persistence_count = 0
            continue
            
        midi_note = round(freq_to_midi_note(freq))
        
        if midi_note == current_note:
            persistence_count += 1
            if persistence_count == persistence_frames:
                persistent_notes.append(midi_note)
        else:
            current_note = midi_note
            persistence_count = 1
    
    return persistent_notes

def analyze_file(file_path):
    try:
        
        with open(file_path, 'r') as f:
            frequencies = [float(line.strip()) for line in f if line.strip()]
        
        persistent_notes = get_persistent_notes(frequencies)
        
        if not persistent_notes:
            return None
            
        return {
            'file': file_path.name,
            'system': str(file_path).replace("/root/yue_pitch_evals/intermediate/", "").split("/")[0],
            'min_note': min(persistent_notes),
            'max_note': max(persistent_notes),
            'range_semitones': max(persistent_notes) - min(persistent_notes)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory(root_dir):
    results = []
    root_path = Path(root_dir)
    
    # Process each file individually
    for file_path in root_path.rglob('*.txt'):
        analysis = analyze_file(file_path)
        if analysis:
            results.append(analysis)
    
    return pd.DataFrame(results)

def create_violin_plot(df, output_path='vocal_ranges.png'):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    sns.violinplot(data=df, x='system', y='range_semitones')
    
    plt.title('Distribution of Vocal Ranges by System')
    plt.xlabel('System')
    plt.ylabel('Range (semitones)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def main():
    root_directory = "raw_pitch_extracted"
    output_file = "vocal_range_analysis.csv"
    plot_file = "vocal_ranges.png"

    print("Processing frequency files...")
    results_df = process_directory(root_directory)
    
    # Save detailed results
    results_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

    print("\nSummary statistics by system:")
    summary = results_df.groupby('system').agg({
        'range_semitones': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    print(summary)
    
    create_violin_plot(results_df, plot_file)
    print(f"\nViolin plot saved to {plot_file}")

if __name__ == "__main__":
    main()