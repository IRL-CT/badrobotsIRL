import pandas as pd
from pydub import AudioSegment
import os
from pathlib import Path

def silence_nodbot(wav_file, csv_file):
    """Silences segments where 'nodbot' is speaking in a WAV file.
    
    Args:
        wav_file: Path to the WAV audio file.
        csv_file: Path to the CSV file containing speaker timings.
    """
    # Try to load the audio file
    try:
        audio = AudioSegment.from_wav(wav_file)
    except FileNotFoundError:
        print(f"Error: WAV file '{wav_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading WAV file '{wav_file}': {str(e)}")
        return

    # Try to load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_file}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: CSV file '{csv_file}' could not be parsed.")
        return

    # Sort the dataframe by start time to ensure proper processing
    df = df.sort_values('start')

    # Create a new audio segment
    modified_audio = audio

    # Process each segment
    for index, row in df.iterrows():
        if row['speaker'].lower() == 'nodbot':
            start_ms = int(row['start'] * 1000)  # Convert seconds to milliseconds
            end_ms = int(row['end'] * 1000)
            
            # Split the audio and replace the segment with silence
            if start_ms > 0:
                before_segment = modified_audio[:start_ms]
            else:
                before_segment = AudioSegment.empty()
                
            if end_ms < len(modified_audio):
                after_segment = modified_audio[end_ms:]
            else:
                after_segment = AudioSegment.empty()
                
            silence = AudioSegment.silent(duration=end_ms - start_ms)
            
            # Concatenate the parts
            modified_audio = before_segment + silence + after_segment
            
            print(f"Silenced segment from {start_ms}ms to {end_ms}ms")

    # Create output directory if it doesn't exist
    output_wav_dir = Path('../../data/wav_silenced')
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    
    # Export the modified audio
    output_wav_file = output_wav_dir / Path(wav_file).name
    modified_audio.export(str(output_wav_file), format="wav")
    print(f"Processed audio saved to: {output_wav_file}")

def process_directory(wav_dir, csv_dir):
    """Process all WAV files in a directory.
    
    Args:
        wav_dir: Directory containing WAV files
        csv_dir: Directory containing corresponding CSV files
    """
    wav_dir = Path(wav_dir)
    csv_dir = Path(csv_dir)
    
    # Process each WAV file
    for wav_file in wav_dir.glob("*.wav"):
        # Look for corresponding CSV file
        csv_file = csv_dir / f"{wav_file.stem}_audio.csv"
        
        if csv_file.exists():
            print(f"\nProcessing {wav_file.name}...")
            silence_nodbot(str(wav_file), str(csv_file))
        else:
            print(f"Warning: CSV file {csv_file} not found for {wav_file}")

if __name__ == "__main__":
    # Example usage for a directory
    wav_directory = "../nodbot_wav_files_silenced"
    csv_directory = "../speaker_diarization_csv"
    process_directory(wav_directory, csv_directory)