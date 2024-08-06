import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, target, sequence_length):
    sequences = [data[i : i + sequence_length] for i in range(len(data) - sequence_length + 1)]
    targets = target[sequence_length - 1 : ]
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def create_data_splits(df, fold_no, fold_col="fold_id", with_val=1, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
        unique_participants = df[df[fold_col] == fold_no]['participant'].unique()
        
        np.random.shuffle(unique_participants)
        
        num_participants = len(unique_participants)
        
        train_size = int(0.7 * num_participants)
        val_size = int(0.2 * num_participants)
        test_size = num_participants - train_size - val_size
                
        train_participants = unique_participants[:train_size]
        val_participants = unique_participants[train_size:train_size + val_size] if with_val == 1 else []
        test_participants = unique_participants[(train_size + val_size):]
        
        participant_to_fold = {}
        
        for participant in train_participants:
            participant_to_fold[participant] = 'train'
        
        for participant in val_participants:
            participant_to_fold[participant] = 'val'
        
        for participant in test_participants:
            participant_to_fold[participant] = 'test'
        
        df[fold_col + '_split'] = df['participant'].map(participant_to_fold)

        train_indices = df[df[fold_col + '_split'] == 'train'].index
        val_indices = df[df[fold_col + '_split'] == 'val'].index if with_val == 1 else []
        test_indices = df[df[fold_col + '_split'] == 'test'].index

        features = df.columns[4:]
        features = [col for col in features if col not in [fold_col, fold_col + '_split']]
        target = df.columns[2]
        
        print(features)
        print(target)

        print(df.loc[train_indices, features].dtypes)


        X_train = df.loc[train_indices, features].astype(np.float32)
        y_train = df.loc[train_indices, target].values.astype(np.float32)
        print(X_train)

        X_val = df.loc[val_indices, features].astype(np.float32) if with_val == 1 else pd.DataFrame()
        y_val = df.loc[val_indices, target].values.astype(np.float32) if with_val == 1 else np.array([])

        X_test = df.loc[test_indices, features].astype(np.float32)
        y_test = df.loc[test_indices, target].values.astype(np.float32)
        
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True) if with_val == 1 else X_val
        X_test = X_test.reset_index(drop=True)
        
        print(f"X_train sample: {X_train.head()}")
        print(f"X_val sample: {X_val.head() if with_val == 1 else 'N/A'}")
        print(f"X_test sample: {X_test.head()}")
        print(f"y_train sample: {y_train[:5]}")
        print(f"y_val sample: {y_val[:5] if with_val == 1 else 'N/A'}")
        print(f"y_test sample: {y_test[:5]}")

        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, sequence_length) if with_val == 1 else (np.array([]), np.array([]))
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, sequence_length)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
df = pd.read_csv("preprocessing/merged_features/all_participants_normalized.csv")

df['fold_id'] = -1

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    df.loc[test_index, 'fold_id'] = fold + 1

with_val = 1
fold_no = 1

splits = create_data_splits(
    df,
    fold_no=fold_no,
    with_val=with_val,
    fold_col='fold_id',
    seed_value=42,
    sequence_length=1
)

if splits is not None:
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences = splits

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Training sequences data shape: {X_train_sequences.shape}, {y_train_sequences.shape}")

    if with_val:
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        print(f"Validation sequences data shape: {X_val_sequences.shape}, {y_val_sequences.shape}")

    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    print(f"Test sequences data shape: {X_test_sequences.shape}, {y_test_sequences.shape}")

else:
    print("Failed to create data splits.")
