import numpy as np
import glob

def freq_to_midi(freq):
    """Convert frequency to MIDI note number"""
    return 12 * np.log2(freq/440) + 69

def get_note_name(midi):
    """Convert MIDI note number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"

def analyze_f0_file(file_path, context_frames=10):
    """
    Analyze F0 file to find max/min values and their surrounding context
    
    Args:
        file_path: Path to the F0 text file
        context_frames: Number of frames to show before and after max/min points
    """
    # Read F0 values
    with open(file_path, 'r') as f:
        f0_values = np.array([float(line.strip()) for line in f if line.strip()])
    
    # Find indices of max and min values
    max_idx = np.argmax(f0_values)
    min_idx = np.argmin(f0_values)
    
    # Convert to MIDI notes for reference
    max_midi = freq_to_midi(f0_values[max_idx])
    min_midi = freq_to_midi(f0_values[min_idx])
    
    # Get context windows
    def get_context(idx):
        start = max(0, idx - context_frames)
        end = min(len(f0_values), idx + context_frames + 1)
        return f0_values[start:end], start
    
    max_context, max_start = get_context(max_idx)
    min_context, min_start = get_context(min_idx)
    
    print(f"\nAnalysis for {file_path}:")
    print("-" * 50)
    print(f"Maximum F0: {f0_values[max_idx]:.2f} Hz ({get_note_name(int(round(max_midi)))}) at frame {max_idx}")
    print(f"Minimum F0: {f0_values[min_idx]:.2f} Hz ({get_note_name(int(round(min_midi)))}) at frame {min_idx}")
    
    print("\nContext around maximum:")
    for i, val in enumerate(max_context):
        frame_idx = max_start + i
        marker = " >> " if frame_idx == max_idx else "    "
        print(f"{marker}Frame {frame_idx}: {val:.2f} Hz")
    
    print("\nContext around minimum:")
    for i, val in enumerate(min_context):
        frame_idx = min_start + i
        marker = " >> " if frame_idx == min_idx else "    "
        print(f"{marker}Frame {frame_idx}: {val:.2f} Hz")

if __name__ == "__main__":
    f0_files = glob.glob("*.txt")
    if not f0_files:
        print("No txt files found in current directory!")
    else:
        for file in sorted(f0_files):
            analyze_f0_file(file)