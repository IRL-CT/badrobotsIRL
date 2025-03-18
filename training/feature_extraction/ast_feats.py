import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple


def extract_features(wav_path, window_duration=1.0, hop_duration=0.5, mel_bins=128):
    """
    Extract mel spectrogram features from an audio file with timeelapseds.
    
    Args:
        wav_path: Path to audio file
        window_duration: Duration of each window in seconds
        hop_duration: Hop size between windows in seconds
        mel_bins: Number of mel bins (default: 128)
    
    Returns:
        List of tuples (features, start_time, end_time)
        features shape: (1024, 128)
    """
    #print('wav path:', wav_path)
    #check if the file exists
    if not Path(wav_path).exists():
        print(f"Warning: File not found: {wav_path}")
        return []
    waveform, sr = torchaudio.load(wav_path)
    
    #print(waveform.shape)
    #print('got here')
    # Resample if necessary
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000
    
    # Calculate window and hop sizes in samples
    window_size = int(window_duration * sr)
    hop_size = int(hop_duration * sr)
    target_length = 1024  # AST expects 1024 time bins
    
    features_list = []
    timestamps = []
    
    # Slide through the audio file
    for start_sample in range(0, waveform.shape[1] - window_size + 1, hop_size):
        # Extract audio segment
        segment = waveform[:, start_sample:start_sample + window_size]
        
        # Calculate timeelapseds
        start_time = start_sample / sr
        end_time = (start_sample + window_size) / sr
        
        # Extract mel spectrogram
        fbank = torchaudio.compliance.kaldi.fbank(
            segment,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=mel_bins,
            dither=0.0,
            frame_shift=10
        )
        
        # Ensure we have exactly 1024 time bins
        if fbank.shape[0] < target_length:
            p = target_length - fbank.shape[0]
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif fbank.shape[0] > target_length:
            fbank = fbank[:target_length, :]
        
        # Normalize using AudioSet stats
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        
        features_list.append(fbank)
        timestamps.append((start_time, end_time))

        
    return torch.stack(features_list), timestamps

def get_window_labels(labels_df, start_time, end_time):
    """
    Get the most common label in a time window.
    
    Args:
        labels_df: DataFrame with timeelapsed and label columns
        start_time: Window start time in seconds
        end_time: Window end time in seconds
    
    Returns:
        Most common label in the window
    """
    window_labels = labels_df[
        (labels_df['timeelapsed'] >= start_time) & 
        (labels_df['timeelapsed'] < end_time)
    ]
    
    if len(window_labels) == 0:
        return None

    #print('size window', window_labels.shape, start_time, end_time)
    
    return window_labels['groundtruth'].iloc[0]


def get_window_features(features_list, timestamps, labels_df):
    """
    Get the latest features in a time window.    
    Args:
        labels_df: DataFrame with timeelapsed and label columns
        start_time: Window start time in seconds
        end_time: Window end time in seconds
    
    Returns:
        Closest features to the window
    """
    #turn features list into a dataframe
    start_times = [x[0] for x in timestamps]
    end_times = [x[1] for x in timestamps]
    
    new_features = []
    for ind, row in labels_df.iterrows():
        time = row['frame']/30.0
        #get feature whose end_time is right after the time
        #look for index in end_times that is greater than time
        ind_time = np.argmax(np.array(end_times) > time)
        

        feature = features_list[ind_time]
        if len(feature) == 0:
            print('no feature found for time:', time, labels_df['participant'].iloc[0])
            #print(labels_df['participant'].iloc[0], time, labels_df['frame'].iloc[-1], features_df['end_time'].iloc[-1])

            continue
        new_features.append(feature)

    #stack all the features by their first dimension
    return torch.stack(new_features)
    
    #if len(window_labels) == 0:
    #    return None

    #print('size window', window_labels.shape, start_time, end_time)
    
    #return window_labels['groundtruth'].iloc[0]

def process_audio_files(audio_dir, label_dir, output_dir, window_duration=1.0, hop_duration=0.5):
    """
    Process all audio files in a directory and save features with labels to CSV.
    
    Args:
        audio_dir: Directory containing audio files
        label_dir: Directory containing label CSV files
        output_dir: Directory to save feature CSVs
        window_duration: Duration of each window in seconds
        hop_duration: Hop size between windows in seconds
    """
    audio_dir = Path(audio_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio file
    for audio_file in tqdm(list(audio_dir.glob('*.wav'))):
        session_id = audio_file.stem
        label_file = label_dir
        
        if not label_file.exists():
            print(f"Warning: No label file found for {audio_file}")
            continue
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        #get only the part that corresponds to the session
        #now, get session id. if audio.wav (audio) is an even number, then session id is str(audio) + str(audio+1) + _0, otherwise it's str(audio-1) + str(audio) + _1
        audio_num = int(audio_file.stem)
        #print(audio_num, 'audio num')
        if audio_num % 2 == 0:
            session_id = str(audio_num) + str(audio_num + 1) + '_0'
        else:
            session_id = str(audio_num - 1) + str(audio_num) + '_1'
        
        #print(session_id, 'session id')

        labels_df = labels_df[labels_df['session'] == session_id]
        
        # Extract features
        features_list, timestamps = extract_features(
            audio_file,
            window_duration=window_duration,
            hop_duration=hop_duration
        )
        
        # Prepare data for DataFrame
        rows = []
        for ind, features in enumerate(features_list):
            start_time, end_time = timestamps[ind]

            # Get label for this window
            label = get_window_labels(labels_df, start_time, end_time)
            
            if label is None:
                print(f"Warning: No labels found for window {start_time}-{end_time} in {session_id}")
                continue
            
            # Flatten the features array
            #print('FEATURE SHAPE:', features.shape)
            flat_features = features.flatten() #keep in mind the original shape is 1024x218
            
            # Create row with timeelapsed, session info, and label
            row = {
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'label': label
            }
            
            # Add features as columns
            for i, value in enumerate(flat_features):
                row[f'feature_{i}'] = value
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        output_file = output_dir / f"{session_id}_features.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved features to {output_file}")


def extract_session_features(audio_path: str, label_df: pd.DataFrame, window_duration: float = 1.0, 
                           hop_duration: float = 0.5, mel_bins: int = 128) -> torch.Tensor:
    """
    Extract AST features for a single session.
    
    Args:
        audio_path: Path to audio file
        window_duration: Duration of each window in seconds
        hop_duration: Hop size between windows in seconds
        mel_bins: Number of mel bins
    
    Returns:
        features: Tensor of shape (num_windows, 1024, 128)
        timestamps: List of start times for each window
    """

    #audio_dir = Path(audio_dir)
    audio_file = Path(audio_path)

    
        
    # Load labels
    #labels_df = labels_df[labels_df['participant'] == session_id]
    # Extract features
    features_list, timestamps = extract_features(
        audio_file,
        window_duration=window_duration,
        hop_duration=hop_duration
    )



    #get the features that correspond to the labels
    features = get_window_features(features_list, timestamps, label_df)
    

    return features


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract AST features from audio files with labels')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing label CSV files')
    parser.add_argument('--session', type=str, required=True,
                        help='Directory containing label CSV files')
    parser.add_argument('--window_duration', type=float, default=1.0,
                        help='Duration of each window in seconds')
    parser.add_argument('--hop_duration', type=float, default=0.5,
                        help='Hop size between windows in seconds')
    
    args = parser.parse_args()
    
    label_df = pd.read_csv(args.label_dir)
    label_df = label_df[label_df['participant'] == args.session]
    
    extract_session_features(
        args.audio_dir,
        label_df,
        args.window_duration,
        args.hop_duration
    )


    #DEFAULT COMMAND
    #python ast_feats.py --audio_dir ../../data/audio/wav/ --label_dir ../../data/all_data_05.csv --output_dir ../../data/ast_feats/ --window_duration 1.0 --hop_duration 0.5