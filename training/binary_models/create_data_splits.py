import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
Generates and returns:
- an array of sequences of data
- an array of corresponding target values

Requires:
- an array of input data that will be used to create sequences 
- an array of target values corresponding to the data
- an array of ids or sessions that will be used to group sequences
- an integer equal to the length of each sequence that will be created.
'''

def create_sequences(data, target, sessions, sequence_length):
    sequences = []
    targets = []

    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]

        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i : i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])
    
    return np.array(sequences), np.array(targets)


def create_sequences_intraparticipant(data, target, sequence_length):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
        targets.append(target[i + sequence_length - 1])
    
    return np.array(sequences), np.array(targets)

'''
def create_data_splits_intraparticipant_binary(df, participant_id, sequence_length=1, neutral_split_ratio=0.8, seed=42):
    try:
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        binary_labels = participant_data["binary_label"].values.astype(int)

        # Split neutral (label 0) into train/test
        neutral_indices = participant_data[participant_data["multiclass_label"] == 0].index.to_numpy()
        np.random.shuffle(neutral_indices)

        split_point = int(len(neutral_indices) * neutral_split_ratio)
        seen_neutral_indices = neutral_indices[:split_point]   # 80% → train
        unseen_neutral_indices = neutral_indices[split_point:] # 20% → test

        first_error_indices = participant_data[participant_data["multiclass_label"] == 1].index
        subsequent_error_indices = participant_data[participant_data["multiclass_label"].isin([2, 3])].index

        train_indices = np.concatenate([first_error_indices, seen_neutral_indices])
        test_indices = np.concatenate([subsequent_error_indices, unseen_neutral_indices])

        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Participant {participant_id}: Empty train or test split. Skipping.")
            return None

        # Extract initial training data
        X_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = binary_labels[train_indices]

        # Split train into train/val (stratified)
        train_df = X_train.copy()
        train_df["label"] = y_train

        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed, stratify=train_df["label"])

        X_train = train_df.drop("label", axis=1).reset_index(drop=True)
        y_train = train_df["label"].values

        X_val = val_df.drop("label", axis=1).reset_index(drop=True)
        y_val = val_df["label"].values

        # Test set
        X_test = features.loc[test_indices].reset_index(drop=True)
        y_test = binary_labels[test_indices]

        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None

        print(f"Participant {participant_id}: Train: {len(train_indices)}, Val: {len(val_df)}, Test: {len(test_indices)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )

    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None
'''

'''
Creates and returns train, val, test splits for a single participant for binary classification based on the binary labels 0, 1

Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the participant id
- an integer equal to the length of each sequence that will be created.
- a float equal to the ratio of neutral samples in the training set
- a float equal to the ratio of error samples in the training set
- an integer seed value for random number generator

Returns:
- X_train: training set data
- X_val: validation set data
- X_test: testing set data
- y_train: training set targets
- y_val: validation set targets
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- X_val_sequences: sequences generated from validation set data
- y_val_sequences: an array of corresponding target values
- X_test_sequences: sequences generated from test set data
- y_test_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''
def create_data_splits_intraparticipant_binary(df, participant_id, sequence_length=1, error_sample_ratio=0.8, seed=42):
    try:

        print("Classic binary split for participant:", participant_id)
        np.random.seed(seed)

        neutral_split_ratio = 0.8

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        binary_labels = participant_data["binary_label"].values.astype(int)

        neutral_indices = participant_data[participant_data["binary_label"] == 0].index.to_numpy()
        error_indices = participant_data[participant_data["binary_label"] == 1].index.to_numpy()
        np.random.shuffle(neutral_indices)
        np.random.shuffle(error_indices)

        neutral_split_point = int(len(neutral_indices) * neutral_split_ratio)
        neutral_train_indices = neutral_indices[:neutral_split_point]   # 80% train
        neutral_test_indices = neutral_indices[neutral_split_point:] # 20% test

        error_split_point = int(len(error_indices) * error_sample_ratio)
        error_train_indices = error_indices[:error_split_point]   # 80% train
        error_test_indices = error_indices[error_split_point:] # 20% test

        train_indices = np.concatenate([neutral_train_indices, error_train_indices])
        test_indices = np.concatenate([neutral_test_indices, error_test_indices])
    
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Participant {participant_id}: Empty train or test split. Skipping.")
            return None

        X_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = binary_labels[train_indices]

        train_df = X_train.copy()
        train_df["label"] = y_train

        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df["label"] if len(train_df["label"].unique()) > 1 else None
        )

        X_train = train_df.drop("label", axis=1).reset_index(drop=True)
        y_train = train_df["label"].values

        X_val = val_df.drop("label", axis=1).reset_index(drop=True)
        y_val = val_df["label"].values

        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = binary_labels[test_indices]

        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None
        
        print(f"Participant {participant_id}: Train: {len(train_indices)}, Val: {len(val_df)}, Test: {len(test_indices)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )
    
    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None


'''
Creates and returns train, val, test splits for a single participant for multiclass classification based on the multiclass labels 1, 2, 3

Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the participant id
- an integer equal to the length of each sequence that will be created.
- a float equal to the ratio of error samples in the training set
- an integer seed value for random number generator
Returns:
- X_train: training set data
- X_val: validation set data
- X_test: testing set data
- y_train: training set targets
- y_val: validation set targets
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- X_val_sequences: sequences generated from validation set data
- y_val_sequences: an array of corresponding target values
- X_test_sequences: sequences generated from test set data
- y_test_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''

def create_data_splits_intraparticipant_multiclass_exclude_neutral(df, participant_id, sequence_length=1, seed=42):
    try:
        print("Multiclass split exclude neutral for participant:", participant_id)
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        labels = participant_data["multiclass_label"].values.astype(int)

        train_indices = []
        test_indices = []

        # Sample from each error label (1, 2, 3)
        for label in [1, 2, 3]:
            label_indices = participant_data[participant_data["multiclass_label"] == label].index.to_numpy()
            if len(label_indices) == 0:
                continue  # skip if no data for this class

            np.random.shuffle(label_indices)
            split_point = int(len(label_indices) * 0.2)

            train_indices.extend(label_indices[:split_point])
            test_indices.extend(label_indices[split_point:])

        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Participant {participant_id}: Empty train or test split. Skipping.")
            return None

        # Prepare training data
        X_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = labels[train_indices]

        # Split off 10% of training data into validation set
        train_df = X_train.copy()
        train_df["label"] = y_train

        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df["label"] if len(train_df["label"].unique()) > 1 else None
        )

        X_train = train_df.drop("label", axis=1).reset_index(drop=True)
        y_train = train_df["label"].values

        X_val = val_df.drop("label", axis=1).reset_index(drop=True)
        y_val = val_df["label"].values

        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = labels[test_indices]

        # Create sequences
        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None
        
        print(f"Participant {participant_id}: Train: {len(train_indices)}, Val: {len(val_df)}, Test: {len(test_indices)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )

    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None


'''
Creates and returns train, val, test splits for a single participant for multiclass classification based on the multiclass labels 0, 1, 2, 3

Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the participant id
- an integer equal to the length of each sequence that will be created.
- a float equal to the ratio of error samples in the training set
- an integer seed value for random number generator
Returns:
- X_train: training set data
- X_val: validation set data
- X_test: testing set data
- y_train: training set targets
- y_val: validation set targets
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- X_val_sequences: sequences generated from validation set data
- y_val_sequences: an array of corresponding target values
- X_test_sequences: sequences generated from test set data
- y_test_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''

def create_data_splits_intraparticipant_multiclass(df, participant_id, sequence_length=1, seed=42):
    try:
        print("Classic multiclass split for participant:", participant_id)
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        labels = participant_data["multiclass_label"].values.astype(int)

        train_indices = []
        test_indices = []

        # use neutral data (label 0) along with error labels (1, 2, 3)
        for label in [0, 1, 2, 3]:
            label_indices = participant_data[participant_data["multiclass_label"] == label].index.to_numpy()
            if len(label_indices) == 0:
                continue

            np.random.shuffle(label_indices)
            split_point = int(len(label_indices) * 0.2)

            train_indices.extend(label_indices[:split_point])
            test_indices.extend(label_indices[split_point:])

        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Participant {participant_id}: Empty train or test split. Skipping.")
            return None

        X_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = labels[train_indices]

        # validation = 10% training
        train_df = X_train.copy()
        train_df["label"] = y_train

        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df["label"] if len(train_df["label"].unique()) > 1 else None
        )

        X_train = train_df.drop("label", axis=1).reset_index(drop=True)
        y_train = train_df["label"].values

        X_val = val_df.drop("label", axis=1).reset_index(drop=True)
        y_val = val_df["label"].values

        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = labels[test_indices]

        # Create sequences
        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None

        print(f"Participant {participant_id}: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_indices)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )

    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None


'''
Creates and returns train, val, test splits for a single participant for binary classification based on a hybrid of binary and multiclass labels.
The function uses multiclass labels to form binary labels.
The training set consists of the first error label (1) and neutral label (0), while the test set consists of the second and third error labels (2, 3) and neutral label (0).

Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the participant id
- an integer equal to the length of each sequence that will be created.
- a float equal to the ratio of error samples in the training set
- an integer seed value for random number generator
Returns:
- X_train: training set data
- X_val: validation set data
- X_test: testing set data
- y_train: training set targets
- y_val: validation set targets
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- X_val_sequences: sequences generated from validation set data
- y_val_sequences: an array of corresponding target values
- X_test_sequences: sequences generated from test set data
- y_test_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''
def create_data_splits_intraparticipant_multiclass_to_binary(df, participant_id, sequence_length=1, seed=42):
    try:
        print("Multiclass to binary split for participant:", participant_id)
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        labels = participant_data["multiclass_label"].values.astype(int)

        label0_indices = participant_data[participant_data["multiclass_label"] == 0].index.to_numpy()
        label1_indices = participant_data[participant_data["multiclass_label"] == 1].index.to_numpy()
        label2_indices = participant_data[participant_data["multiclass_label"] == 2].index.to_numpy()
        label3_indices = participant_data[participant_data["multiclass_label"] == 3].index.to_numpy()

        np.random.shuffle(label1_indices)
        np.random.shuffle(label0_indices)
        np.random.shuffle(label2_indices)
        np.random.shuffle(label3_indices)

        # downsample label 0 to match label 1 count and prevent data imbalance
        if len(label0_indices) >= len(label1_indices):
            label0_train_indices = label0_indices[:len(label1_indices)]
            label0_test_indices = label0_indices[len(label1_indices):]
        else:
            label0_train_indices = label0_indices
            label0_test_indices = np.array([])

        train_indices = np.concatenate([label1_indices, label0_train_indices])
        test_indices = np.concatenate([label0_test_indices, label2_indices, label3_indices])

        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Participant {participant_id}: Empty train or test split. Skipping.")
            return None
        
        X_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = np.array([1 if labels[i] == 1 else 0 for i in train_indices])

        train_df = X_train.copy()
        train_df["label"] = y_train

        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df["label"] if len(np.unique(y_train)) > 1 else None
        )

        X_train = train_df.drop("label", axis=1).reset_index(drop=True)
        y_train = train_df["label"].values

        X_val = val_df.drop("label", axis=1).reset_index(drop=True)
        y_val = val_df["label"].values

        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = np.array([1 if labels[i] in [2, 3] else 0 for i in test_indices])

        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None
        
        print(f"Participant {participant_id}: Train: {len(train_indices)}, Val: {len(val_df)}, Test: {len(test_indices)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )

    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None


'''
Creates and returns train, val, test splits for a single participant for binary classification based on a hybrid of binary and multiclass labels.
The function uses both binary and multiclass labels to form binary labels.
The training set consists of the first error label (1) and neutral label (0) from the binary labels, while the test set consists of unseen error labels (2, 3) and unseen neutral label (0) from the multiclass labels.
Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the participant id
- an integer equal to the length of each sequence that will be created.
- an integer seed value for random number generator
Returns:
- X_train: training set data
- X_val: validation set data
- X_test: testing set data
- y_train: training set targets
- y_val: validation set targets
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- X_val_sequences: sequences generated from validation set data
- y_val_sequences: an array of corresponding target values
- X_test_sequences: sequences generated from test set data
- y_test_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned

'''
def create_data_splits_intraparticipant_hybrid_binary(df, participant_id, sequence_length=1, seed=42):
    try:
        print("Hybrid binary split for participant:", participant_id)
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        binary_labels = participant_data["binary_label"].values.astype(int)
        multiclass_labels = participant_data["multiclass_label"].values.astype(int)

        label0_indices = participant_data[participant_data["multiclass_label"] == 0].index.to_numpy()
        label1_indices = participant_data[participant_data["multiclass_label"] == 1].index.to_numpy()
        label2_indices = participant_data[participant_data["multiclass_label"] == 2].index.to_numpy()
        label3_indices = participant_data[participant_data["multiclass_label"] == 3].index.to_numpy()

        np.random.shuffle(label1_indices)
        np.random.shuffle(label0_indices)
        np.random.shuffle(label2_indices)
        np.random.shuffle(label3_indices)

        label0_train, label0_test = train_test_split(label0_indices, test_size=0.2, random_state=seed)
        label1_train, label1_test = train_test_split(label1_indices, test_size=0.2, random_state=seed) # label1_test will not be used
        label2_train, label2_test = train_test_split(label2_indices, test_size=0.2, random_state=seed)
        label3_train, label3_test = train_test_split(label3_indices, test_size=0.2, random_state=seed)

        # create binary labels for training set. 1, 2, 3 labels from multiclass_labels are considered as 1, while 0 is considered as 0
        train_indices = np.concatenate([label0_train, label1_train, label2_train, label3_train])
        y_train_binary = np.array([0 if multiclass_labels[i] == 0 else 1 for i in train_indices]) 

        # val split
        X_train_full = features.iloc[train_indices].reset_index(drop=True)
        y_train_full = y_train_binary
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.1,
            random_state=seed,
            stratify=y_train_full
        )

        # create binary labels for test set. 2, 3 labels from multiclass_labels are considered as 1, while 0 is considered as 0
        test_indices = np.concatenate([label0_test, label2_test, label3_test])
        y_test_binary = np.array([0 if multiclass_labels[i] == 0 else 1 for i in test_indices])

        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = y_test_binary

        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None
        
        print(f"Participant {participant_id}: Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        print(f"Label distribution: Train {np.bincount(y_train)}, Val {np.bincount(y_val)}, Test {np.bincount(y_test)}")

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            sequence_length
        )

    except Exception as e:
        print(f"An error occurred for participant {participant_id}: {e}")
        return None
       

'''
Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the number of folds to create for cross validation
- the index of the fold to be used for the current train validation test split
- an integer seed value for random number generator
- an integer equal to the length of each sequence that will be created.

Creates and returns:
- X_train: training set data
- y_train: training set targets
- X_val: validation set data
- y_val: validation set targets
- X_test: testing set data
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''
def create_data_splits(df, model, fold_no, num_folds=5, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        if model == "multiclass":
            target_col = 3
        elif model == "binary":
            target_col = 2

        features = df.iloc[:, 4:]
        target = df.iloc[:, target_col].values.astype('int')
        sessions = df['participant'].values
        
        fold_sessions = df['participant'].unique()
        num_of_sessions = len(fold_sessions)

        if num_of_sessions < num_folds:
            raise ValueError("Number of sessions is less than the number of folds. Adjust the number of folds.")
    
        # 70-20-10 train-val-test split, make sure at least 1 sample per split
        train_size = int(np.floor(0.7 * num_of_sessions))
        val_size = int(np.ceil(0.2 * num_of_sessions))
        test_size = num_of_sessions - train_size - val_size

        np.random.shuffle(fold_sessions)

        train_folds = []
        val_folds = []
        test_folds = []

        for i in range(num_folds):

            start_train_index = i * val_size
            end_train_index = (start_train_index + train_size if start_train_index+train_size <= len(fold_sessions) else start_train_index +  train_size - len(fold_sessions))

            if start_train_index >= end_train_index:
                train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
            else:
                train_fold = fold_sessions[start_train_index : end_train_index]

            val_train_index = end_train_index
            val_end_index = (val_train_index + val_size if val_train_index+val_size <= len(fold_sessions) else val_train_index +  val_size - len(fold_sessions))

            if val_train_index >= val_end_index:
                val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
            else:
                val_fold = fold_sessions[val_train_index : val_end_index]

            test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))

            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)

        train_fold = train_folds[fold_no]
        val_fold = val_folds[fold_no]
        test_fold = test_folds[fold_no]

        print("Train fold:", train_fold)
        print("Validation fold:", val_fold)
        print("Test fold:", test_fold)

        train_indices = df[df['participant'].isin(train_fold)].index
        val_indices = df[df['participant'].isin(val_fold)].index
        test_indices = df[df['participant'].isin(test_fold)].index

        if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
            print(f"One of the folds is empty (fold {fold_no}). Skipping this fold.")
            return None

        X_train = features.loc[train_indices]
        y_train = target[train_indices]
        session_train = sessions[train_indices]
        print("Train shapes:", X_train.shape, y_train.shape)

        X_val = features.loc[val_indices]
        y_val = target[val_indices]
        session_val = sessions[val_indices]
        print("Validation shapes:", X_val.shape, y_val.shape)

        X_test = features.loc[test_indices]
        y_test = target[test_indices]
        session_test = sessions[test_indices]
        print("Test shapes:", X_test.shape, y_test.shape)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)

        if len(X_train_sequences) == 0 or len(X_val_sequences) == 0 or len(X_test_sequences) == 0:
            print(f"Sequences for fold {fold_no} are empty. Skipping this fold.")
            return None

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

'''
Requires:
- a dataframe (to be reduced via principal component analysis) consisting of features to be trained on and target values
- an integer equal to the number of folds to create for cross validation
- the index of the fold to be used for the current train validation test split
- an integer seed value for random number generator
- an integer equal to the length of each sequence that will be created.

Implements principal component analysis and returns:
- X_train: training set data
- y_train: training set targets
- X_val: validation set data
- y_val: validation set targets
- X_test: testing set data
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''

def create_data_splits_pca(df, model, fold_no, num_folds=5, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        participant_frames_labels = df.iloc[:, :4]

        if model == "multiclass":
            target_col = 3
        elif model == "binary":
            target_col = 2

        x = df.iloc[:, 4:]
        x = StandardScaler().fit_transform(x.values)

        pca = PCA(n_components=0.90)
        principal_components = pca.fit_transform(x)
        print(principal_components.shape)

        principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
        principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

        df = principal_df
        #df.to_csv("principal_df.csv", index=False)

        print(df)

        features = df.iloc[:, 4:]
        target = df.iloc[:, target_col].values.astype('int')
        sessions = df['participant'].values
        
        fold_sessions = df['participant'].unique()
        num_of_sessions = len(fold_sessions)

        if num_of_sessions < num_folds:
            raise ValueError("Number of sessions is less than the number of folds. Adjust the number of folds.")
    
        # 70-20-10 train-val-test split, make sure at least 1 sample per split
        train_size = int(np.floor(0.7 * num_of_sessions))
        val_size = int(np.ceil(0.2 * num_of_sessions))
        test_size = num_of_sessions - train_size - val_size

        np.random.shuffle(fold_sessions)

        train_folds = []
        val_folds = []
        test_folds = []

        for i in range(num_folds):

            start_train_index = i * val_size
            end_train_index = (start_train_index + train_size if start_train_index+train_size <= len(fold_sessions) else start_train_index +  train_size - len(fold_sessions))

            if start_train_index >= end_train_index:
                train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
            else:
                train_fold = fold_sessions[start_train_index : end_train_index]

            val_train_index = end_train_index
            val_end_index = (val_train_index + val_size if val_train_index+val_size <= len(fold_sessions) else val_train_index +  val_size - len(fold_sessions))

            if val_train_index >= val_end_index:
                val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
            else:
                val_fold = fold_sessions[val_train_index : val_end_index]

            test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))

            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)

        train_fold = train_folds[fold_no]
        val_fold = val_folds[fold_no]
        test_fold = test_folds[fold_no]

        print("Train fold:", train_fold)
        print("Validation fold:", val_fold)
        print("Test fold:", test_fold)

        train_indices = df[df['participant'].isin(train_fold)].index
        val_indices = df[df['participant'].isin(val_fold)].index
        test_indices = df[df['participant'].isin(test_fold)].index

        if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
            print(f"One of the folds is empty (fold {fold_no}). Skipping this fold.")
            return None

        X_train = features.loc[train_indices]
        y_train = target[train_indices]
        session_train = sessions[train_indices]
        print("Train shapes:", X_train.shape, y_train.shape)

        X_val = features.loc[val_indices]
        y_val = target[val_indices]
        session_val = sessions[val_indices]
        print("Validation shapes:", X_val.shape, y_val.shape)

        X_test = features.loc[test_indices]
        y_test = target[test_indices]
        session_test = sessions[test_indices]
        print("Test shapes:", X_test.shape, y_test.shape)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)

        if len(X_train_sequences) == 0 or len(X_val_sequences) == 0 or len(X_test_sequences) == 0:
            print(f"Sequences for fold {fold_no} are empty. Skipping this fold.")
            return None

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_data_splits_intra(df, fold_no=0, train_ratio=0.05, test_ratio=0.20, seed_value=42, sequence_length=1):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 3:]
        target = df.iloc[:, 2].values.astype('int')
        sessions = df['participant'].values
        
        # Get unique sessions/participants
        unique_sessions = df['participant'].unique()
        
        # Select the session for this fold
        if fold_no >= len(unique_sessions):
            raise ValueError(f"fold_no {fold_no} exceeds number of available sessions {len(unique_sessions)}")
        
        current_session = unique_sessions[fold_no]
        print(f"Processing session: {current_session}")
        
        # Get all data for this session, maintaining chronological order
        session_mask = df['participant'] == current_session
        session_df = df[session_mask].copy()
        session_indices = session_df.index
        
        # Ensure chronological order (assuming data is already sorted, but just to be safe)
        session_df = session_df.sort_index()
        session_indices = session_df.index
        
        n_samples = len(session_df)
        print(f"Total samples in session {current_session}: {n_samples}")
        
        # Calculate split points
        train_size = int(np.floor(train_ratio * n_samples))
        test_size = int(np.floor(test_ratio * n_samples))
        
        # Ensure minimum sizes
        if train_size < 1:
            train_size = 1
        if test_size < 1:
            test_size = 1
            
        # Ensure we don't exceed total samples
        if train_size + test_size >= n_samples:
            # Adjust sizes proportionally
            available = n_samples - 1  # Leave at least 1 for test
            train_size = max(1, int(available * train_ratio / (train_ratio + test_ratio)))
            test_size = max(1, available - train_size)
        
        val_size = n_samples - train_size - test_size
        
        print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create chronological splits
        train_end = train_size
        val_end = train_size + val_size
        
        # Get indices for each split
        train_indices = session_indices[:train_end]
        val_indices = session_indices[train_end:val_end]
        test_indices = session_indices[val_end:]
        
        # Extract features and targets
        X_train = features.loc[train_indices]
        y_train = target[train_indices]
        session_train = sessions[train_indices]
        
        X_val = features.loc[val_indices]
        y_val = target[val_indices]
        session_val = sessions[val_indices]
        
        X_test = features.loc[test_indices]
        y_test = target[test_indices]
        session_test = sessions[test_indices]
        
        print("Train shapes:", X_train.shape, y_train.shape)
        print("Val shapes:", X_val.shape, y_val.shape)
        print("Test shapes:", X_test.shape, y_test.shape)
        
        # Reset indices
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        # Create sequences
        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)
        print("Val sequences shape:", X_val_sequences.shape, y_val_sequences.shape)
        print("Test sequences shape:", X_test_sequences.shape, y_test_sequences.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None