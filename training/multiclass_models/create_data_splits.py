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


def create_data_splits_intraparticipant_binary(df, participant_id, sequence_length=1, neutral_split_ratio=0.8, seed=42):
    try:
        np.random.seed(seed)

        participant_data = df[df["participant"] == participant_id].copy().reset_index(drop=True)
        features = participant_data.iloc[:, 4:]
        labels = participant_data["multiclass_label"].values.astype(int)

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
        y_train = labels[train_indices]

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
        y_test = labels[test_indices]

        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None

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


def create_data_splits_intraparticipant_multiclass(df, participant_id, sequence_length=1, error_sample_ratio=0.2, seed=42):
    try:
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
            split_point = int(len(label_indices) * error_sample_ratio)

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

        # Test set (on the remainder of the error types)
        X_test = features.iloc[test_indices].reset_index(drop=True)
        y_test = labels[test_indices]

        # Create sequences
        X_train_seq, y_train_seq = create_sequences_intraparticipant(X_train.values, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences_intraparticipant(X_val.values, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences_intraparticipant(X_test.values, y_test, sequence_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Participant {participant_id}: Empty sequence data. Skipping.")
            return None

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

