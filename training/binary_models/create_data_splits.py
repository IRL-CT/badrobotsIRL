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

def create_data_splits(df, fold_no, num_folds=5, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 4:]
        target = df.iloc[:, 2].values.astype('int')
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

def create_multiclass_data_splits(df, fold_no, num_folds=5, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 4:]
        target = df.iloc[:, 3].values.astype('int')
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

def create_data_splits_pca(df, fold_no, num_folds=5, seed_value=42, sequence_length=1):
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        participant_frames_labels = df.iloc[:, :4]

        x = df.iloc[:, 4:]
        x = StandardScaler().fit_transform(x.values)
        pca = PCA()
        principal_components = pca.fit_transform(x)
        print(principal_components.shape)

        pca = PCA(n_components=0.90)
        principal_components = pca.fit_transform(x)
        print(principal_components.shape)

        principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
        principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

        df = principal_df
        df.to_csv("principal_df.csv", index=False)

        print(df)

        features = df.iloc[:, 4:]
        target = df.iloc[:, 2].values.astype('int')
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

