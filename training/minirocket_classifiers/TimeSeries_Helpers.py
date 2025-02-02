from tsai.all import *
import numpy as np
from fastai.callback.wandb import *
import wandb
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from create_data_splits import create_data_splits, create_data_splits_ids
from get_metrics import get_metrics

from fastai.callback.core import Callback
from fastai.metrics import accuracy, Precision, Recall, F1Score
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.utils import resample


class MetricsCallback(Callback):

    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.valid_f1s = []
        self.valid_precisions = []
        self.valid_recalls = []


    def after_fit(self):
        "Store the metrics after each epoch"
        # Store training metrics
        self.values.append(self.learn.recorder.values)

def most_common(lst):
    '''
    Returns the most common element in a list
    '''
    data = Counter(lst)
    return max(lst, key=data.get)




def create_single_label_per_interval_with_context(df, interval_length, stride, features, context_length, balanced=False):
    '''
    Splits data into intervals of length interval_length + contextlength. The label is based only on the last frames of length interval_length
    :param df: dataframe containing the gaze data and labels
    :param interval_length: length of one of the resulting samples
    :param stride: by how many frames the next interval sample is moved. Non-overlaping if stride==interval_length
    :param features: list of features used
    :param context_length: length of the additional context
    :return: labeled interval data and original labels (not cut into intervals with majority-based label)
    '''
    labels = np.empty(0)
    values = []
    print('df columns in data prep', df.columns)    
    
    #print if df has nan values
    for i in range(0, len(df), stride):
        if i + interval_length + context_length <= len(df):
            interval_labels = list(
                df["binary_label"][i+context_length:i+context_length+interval_length])
            #print('interval labels', interval_labels)
            majority_label =  interval_labels[-1] #most_common(interval_labels)
            labels = np.append(labels, majority_label)
            # determine sample values (sample is 2d array of features, data)
            sample = []
            for feat in df.columns[4:]:
                #print('feature here', feat)
                sample.append(
                        list(df[feat][i:i+interval_length+context_length]))
        
            values.append(sample)

    #print('labels', labels)
    # Count class distribution
    class_counts = Counter(labels)
    print("Original class distribution:", class_counts)
    
    # Determine minority class count
    minority_class_count = min(class_counts.values())
    
    # Balance dataset
    balanced_values = []
    balanced_labels = []
    
    for clss in class_counts:
        class_mask = [l == clss for l in labels]
        class_samples = [values[j] for j in range(len(values)) if class_mask[j]]
        class_labels = [labels[j] for j in range(len(labels)) if class_mask[j]]
        
        # Randomly select samples
        resampled_indices = np.random.choice(
            len(class_samples), 
            size=minority_class_count, 
            replace=False
        )
        
        balanced_values.extend([class_samples[idx] for idx in resampled_indices])
        balanced_labels.extend([class_labels[idx] for idx in resampled_indices])

    #PRINT first element of balanced values
    #print('BALANCED VALUES', balanced_values[0])
    
    print("Balanced class distribution:", Counter(balanced_labels))

    if balanced:
        return balanced_values, balanced_labels, df["binary_label"]
    else:
        return values, labels, df["binary_label"]

# Select Modelities
def modalities_combination_data_prep(modalities_combination_vec, X_train, feature_set_tag): # groundtruth):
    selected_modalities_train = pd.DataFrame()
    #print(X_train.columns)

    if (feature_set_tag == 'Stat'):
        stat_feature_df = pd.read_csv("stats_features_ttest_full.csv")
        stat_features = stat_feature_df['feature'].tolist()
        # if groundtruth == 'stat':
        #     stat_feature_df = pd.read_csv("../../data/sign_features_05.csv")
        #     stat_features = stat_feature_df['feature'].tolist()
        # else:
        #     stat_feature_df = pd.read_csv("../../data/sign_features_sign05.csv")
        #     stat_features = stat_feature_df['feature'].tolist()
    elif (feature_set_tag == 'RF'):
        rf_feature_df = pd.read_csv("rf_features_selected_40.csv")
        impurity_features = rf_feature_df['feature'].tolist()
        # if groundtruth == 'sign':
        #     rf_feature_df = pd.read_csv("../../data/rf_top_sign05.csv")
        #     impurity_features = rf_feature_df['Feature'].tolist()
        
    
    if modalities_combination_vec[0]: # audio
        cols_or =  selected_modalities_train.columns
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 63:88]], axis=1, ignore_index=True)

        # selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 59:123]], axis=1, ignore_index=True)
        # print(selected_modalities_train.shape)
        # selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 177:241]], axis=1, ignore_index=True)
        print(selected_modalities_train.shape)

        #column names, concatenate 2 lists
        #cols_names = X_train.columns[59:123] 
        #cols_names = cols_names.append(X_train.columns[177:241])
        cols_names = X_train.columns[63:88] 

        print('COLS NAMES', cols_names)
        cols = cols_or.append(cols_names)
        selected_modalities_train.columns = cols
        #print(selected_modalities_train.head())
        print('ADDED AUDIO')
    if modalities_combination_vec[1]: # face
        cols_or =  selected_modalities_train.columns
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 28:63]], axis=1, ignore_index=True)
        print(selected_modalities_train.shape)
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 88:]], axis=1, ignore_index=True)
        print(selected_modalities_train.shape)
        cols_names = X_train.columns[28:63]
        cols_names = cols_names.append(X_train.columns[88:])
        print('COLS NAMES', cols_names)

        cols = cols_or.append(cols_names)

        selected_modalities_train.columns = cols
        #print(selected_modalities_train.head())
        print('ADDED FACE')
    if modalities_combination_vec[2]: # talk
        cols_or =  selected_modalities_train.columns
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 4:28]], axis=1, ignore_index=True)
        print(selected_modalities_train.shape)
        cols_names = X_train.columns[4:28]
        #make it index
        cols_names = pd.Index(cols_names)
        print('COLS NAMES', cols_names)
        cols = cols_or.append(cols_names)
        selected_modalities_train.columns = cols
        #print(selected_modalities_train.head())

        print('ADDED POSE')

    
    cols = selected_modalities_train.columns

       
    if feature_set_tag == "Stat":
        new_cols = [f for f in cols if f in stat_features]
        #('new cols', new_cols)
        #print(new_cols)
        #now, select only the columns that are in the stat_features
        new_selected_modalities_train = selected_modalities_train.loc[:, new_cols]
        print(new_selected_modalities_train.shape)
        selected_modalities_train = new_selected_modalities_train
        print('ADDED STAT')
        print(selected_modalities_train.columns)
    elif feature_set_tag == "RF": #excluding MULTI
        new_cols = [f for f in cols if f not in impurity_features]
        #print(new_cols)
        #now, select only the columns that are in the stat_features
        new_selected_modalities_train = selected_modalities_train.loc[:, new_cols]
        selected_modalities_train = new_selected_modalities_train
        print('ADDED RF')
        print(selected_modalities_train.columns)
    else:
        new_cols = [f for f in cols]
        new_selected_modalities_train = selected_modalities_train.loc[:, new_cols]
        selected_modalities_train = new_selected_modalities_train
        print('ADDED ALL')
        print(selected_modalities_train.columns) 

    

    return selected_modalities_train

def apply_pca(df):
    pca = PCA(n_components=0.9)
    #check if there is nan
    #print('NAN VALUES', df.isnull().sum())
    df_pca = pca.fit_transform(df.iloc[:,4:])
    df_pca = pd.DataFrame(df_pca)
    #reset index
    df_pca = df_pca.reset_index(drop=True)
    df_pca = pd.concat([df.loc[:, ['frame','participant','binary_label','multiclass_label']], df_pca], axis=1, ignore_index=True)
    #create string with pc + number for features
    df_pca.columns = ['frame','participant','binary_label','multiclass_label'] + ['pc' + str(i) for i in range(1, df_pca.shape[1]-2)]
    #remove nan values
    df_pca = df_pca.dropna()

    #df_pca.columns = ['pair_id', 'start_seconds', 'is_discomfort'] + list(df_pca.columns[3:])
    return df_pca


def read_in_data(df_name, threshold, interval_length, stride_train, stride_eval, features, valid_ids, test_ids, context_length, config):
    """
    reads in merged csvs: applies recompute_treshold() and create_single_label_per_interval

    :param threshold: threshold for AoI analysis
    :param interval_length: length of one of the resulting samples
    :param stride_train: by how many frames the next interval sample is moved. Non-overlaping if stride==interval_length
    :param stride_eval: by how many frames the next interval sample is moved (used on val/test samples). Non-overlaping if stride==interval_length
    :param features: list of features used
    :param valid_ids: list of ids used for validation
    :param test_ids: list of ids used for testing
    :param context_length: length of the additional context

    :return: values and labels for lvl1 and lvl2 for all participants
    """
    path = df_name
    data = {}

    
    df = pd.read_csv(path)
    #transform participant ids to be in ascendent order from 0 to n_p
    participants_uniques = df['participant'].unique()
    dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
    #print(dict_participant)
    df['participant'] = df['participant'].map(dict_participant)
    participants_uniques = df['participant'].unique()
    print(participants_uniques)

    feature_modalities = modalities_combination_data_prep(config.modalities_combination, df, config.feature_set_tag) #,config.groundtruth)
    print('FEATURE MODALITIES', feature_modalities.shape)
    #if it's fot no columns, finish compute
    if feature_modalities.shape[1] == 0:
        return None, None

    #print('FEATURE MODALITIES', feature_modalities.columns)
    #select 3 cols of df_p
    init_cols = df.loc[:, ['frame','participant','binary_label','multiclass_label']]
    df = pd.concat([init_cols, feature_modalities], axis=1, ignore_index=True)
    print('new df shape', df.shape)
    #column names
    df.columns = ['frame','participant','binary_label','multiclass_label'] + list(feature_modalities.columns)
    if config.dataset_processing == "pca":
        df = apply_pca(df)
        
    for i in participants_uniques:
        participant = i
        data[participant] = {}
        #print('in', i)
        df_p = df[df["participant"] == i]
        #reindex
        df_p = df_p.reset_index(drop=True)

        data[participant]= df_p
        print(data[participant].shape)

    lvl1_data = []

    for p_number, participant in enumerate(data):
        if participant in valid_ids or participant in test_ids:
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant], interval_length=interval_length, stride=stride_eval, features=features, context_length=context_length, balanced=config.balanced)
            lvl1_data.append((values, labels, raw_labels))
           
        else:
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant], interval_length=interval_length, stride=stride_train, features=features, context_length=context_length, balanced=config.balanced)
            lvl1_data.append((values, labels, raw_labels))
            
    return lvl1_data, data


def dataPrep(df_name, threshold, interval_length, stride_train, stride_eval, train_ids, valid_ids, test_ids, use_lvl1, use_lvl2 , merge_labels, batch_size, batch_tfms, features, context_length, oversampling, undersampling,config, verbose=True):
    '''
    :param threshold: threshold for AoI
    :param interval_length: how many frames make up one datapoint
    :param stride_train: how many frames are skipped between two datapoints (train)
    :param stride_eval: how many frames are skipped between two datapoints (eval)
    :param train_ids: list of participant ids for training
    :param valid_ids: list of participant ids for validation
    :param test_ids: list of participant ids for testing
    :param use_lvl1: boolean indicating if level 1 data is used
    :param use_lvl2: boolean indicating if level 2 data is used
    :param merge_labels: boolean indicating if confusion and error are merged
    :param batch_size: batch size for training
    :param batch_tfms: transformations
    :param features: list of timeseries features used
    :param context_length: length of context
    :param oversampling: boolean indicating if oversampling is used
    :param undersampling: boolean indicating if undersampling is used

    :return: dataloader and individual datasets/labels
    '''

    #print('CHECKING UPON DATA PREP')
    #print('valid_ids', valid_ids)
    #print('test_ids', test_ids)
    lvl1, data = read_in_data(df_name = df_name, threshold=threshold, interval_length=interval_length,
                                    stride_train=stride_train, stride_eval=stride_eval, features=features, valid_ids=valid_ids, test_ids=test_ids, context_length=context_length, config=config)

    #if data is none, return none
    if data is None:
        return None

    #print('lvl1', len(lvl1))
    features = data.get(0).columns[4:]

    # prepare labels (1d array) and data (3D array) for TSAI
    X_train = np.empty(
        (0, len(features), interval_length+context_length), dtype=np.float64)
    X_val = np.empty((0, len(features),
                     interval_length+context_length), dtype=np.float64)
    X_test = np.empty((0, len(features),
                      interval_length+context_length), dtype=np.float64)
    y_train = np.empty(0)
    y_val = np.empty(0)
    y_test = np.empty(0)

    # for final eval after concatenation
    y_test_raw = []
    y_val_raw = []
    X_val_by_participant = []
    X_test_by_participant = []

    #print('use lvl1', use_lvl1)
    #print('train_ids', train_ids)
    #print('valid_ids', valid_ids)
    #print('test_ids', test_ids)
    #print(len(lvl1))

    # merge data of participants based on ids
    for i in train_ids:
        #print(i)
        if use_lvl1:
            y_train = np.append(y_train, lvl1[i][1])
            #print(np.array(lvl1[i][0]).shape)
            #print(np.array(lvl1[i][1]).shape)

            X_train = np.append(X_train, lvl1[i][0], axis=0)
    for i in valid_ids:
        if use_lvl1:
            y_val = np.append(y_val, lvl1[i][1])
            X_val = np.append(X_val, lvl1[i][0], axis=0)
            y_val_raw.append(lvl1[i][2])
            X_p = np.empty((0, len(features),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl1[i][0], axis=0)
            X_val_by_participant.append(X_p)
    for i in test_ids:
        if use_lvl1:
            y_test = np.append(y_test, lvl1[i][1])
            X_test = np.append(X_test, lvl1[i][0], axis=0)
            y_test_raw.append(lvl1[i][2])
            X_p = np.empty((0, len(features),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl1[i][0], axis=0)
            X_test_by_participant.append(X_p)
        

    if oversampling:
        ros = RandomOverSampler(random_state=0, sampling_strategy={0: Counter(y_train)[
                                0], 1: Counter(y_train)[1]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]
    if undersampling:
        ros = RandomUnderSampler(random_state=0, sampling_strategy={"Normal": Counter(y_train)[
                                 "Confusion"], "Confusion": Counter(y_train)["Confusion"], "Error": Counter(y_train)["Error"]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]

    if merge_labels:
        y_train[y_train == "Error"] = "Confusion"
        if len(y_val) > 0:
            y_val[y_val == "Error"] = "Confusion"
        if len(y_test) > 0:
            y_test[y_test == "Error"] = "Confusion"

    if verbose:
        print("Train Labels:", y_train.shape)
        print("Val Labels", y_val.shape)
        print("Test Labels:", y_test.shape)
        print("Train Data:", X_train.shape)
        print("Val Data:", X_val.shape)
        print("Test Data:", X_test.shape)
        #count of lables
        print("Train Labels:", Counter(y_train))
        print("Val Labels:", Counter(y_val))
        print("Test Labels:", Counter(y_test))

    X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])

    tfms = [None, TSClassification()]  # transforms for the data
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[
                                   batch_size, 128], batch_tfms=batch_tfms, num_workers=0)
    print('DATALOADERS')
    print(dls.c)
    return dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant


def vsBaseline(input: Tensor, targs: Tensor):
    "Computes difference of achieved accuracy and baseline accuracy; majority class is hardcoded"
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    accuracy = (input == targs).float().mean()
    # compute baseline as the accuracy of always predicting the majority class
    majority_class = tensor(1)
    baseline_accuracy = (majority_class == targs).float().mean()
    return accuracy-baseline_accuracy


def vsBaseline_merged(input: Tensor, targs: Tensor):
    "Computes difference of achieved accuracy and baseline accuracy given only 2 labels; majority class is hardcoded"
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    accuracy = (input == targs).float().mean()
    # compute baseline as the accuracy of always predicting the majority class
    majority_class = tensor(1)
    baseline_accuracy = (majority_class == targs).float().mean()
    return accuracy-baseline_accuracy


def load_config_model(config, dls, cbs):
    if config.merged_labels:  # baseline metric has hardcoded majority class and needs to know if labels are merged
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro")]
    else:
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro")]
        
    cb_metrics = MetricsCallback()


    if config.model == "InceptionTime":
        model = InceptionTime(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "InceptionTimePlus":
        model = InceptionTimePlus(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "TST":
        model = TST(dls.vars, dls.c, dls.len,
                    dropout=config.dropout_TST, fc_dropout=config.fc_dropout_TST) 
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics,cbs = cbs, loss_func=FocalLossFlat()) 
        else:
            return Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(),  metrics=metrics, cbs=cbs) 
    elif config.model == "XceptionTime":
        model = XceptionTime(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "ResNet":
        model = ResNet(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "xresnet1d34":
        model = xresnet1d34(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "ResCNN":
        model = ResCNN(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "OmniScaleCNN":
        model = OmniScaleCNN(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "mWDN":
        model = mWDN(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "LSTM_FCN":
        model = LSTM_FCN(dls.vars, dls.c, dls.len,
                         fc_dropout=config.fc_dropout_LSTM_FCN, rnn_dropout=config.dropout_LSTM_FCN)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs = cbs,  loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics,cbs = cbs)

    elif config.model == "GRU_FCN":
        model = LSTM_FCN(dls.vars, dls.c, dls.len,
                         fc_dropout=config.fc_dropout_LSTM_FCN, rnn_dropout=config.dropout_LSTM_FCN)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs = cbs,  loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics,cbs = cbs)
    elif config.model == "LSTM":
        model = LSTM(dls.vars, dls.c, n_layers=3, bidirectional=True)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 
    elif config.model == "gMLP":
        model = gMLP(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs) 


def evaluate_preds_against_raw(y_preds_per_participant, y_raw_per_participant, stride, context_length, interval_length, plotting):
    all_y_true_concatenated = []
    all_y_pred_concatenated = []
    for i, y_preds in enumerate(y_preds_per_participant):
        y_preds_concatenated = [y_preds[0]]*(interval_length-stride)
        y_raw = y_raw_per_participant[i]
        for j in range(1, len(y_preds)):
            y_preds_concatenated += ([y_preds[j]]*stride)
        # compute accuracy
        # make y_preds_concatenated as long as y_raw and fill the missing values with the "Normal"
        y_preds_concatenated += [0] * \
            (len(y_raw)-len(y_preds_concatenated))
        accuracy = accuracy_score(
            y_raw, y_preds_concatenated)  # old: y_raw[:len(y_preds_concatenated)]
        baseline = len(y_raw[y_raw == 0])/len(y_raw)
        all_y_true_concatenated += list(y_raw)
        all_y_pred_concatenated += y_preds_concatenated
        if plotting:
            # plot y_raw and y_preds
            print("Accuracy: ", accuracy, "Baseline: ", baseline)
            plt.plot(y_raw[:len(y_preds_concatenated)], label="y_raw")
            plt.plot(y_preds_concatenated, label="y_preds")
            plt.legend()
            plt.show()
    accuracy = accuracy_score(all_y_true_concatenated, all_y_pred_concatenated)
    macro_f1 = f1_score(
        all_y_true_concatenated, all_y_pred_concatenated, average="macro")
    baseline = all_y_true_concatenated.count(
        0) / len(all_y_true_concatenated)
    all_y_true_concatenated = [
        1 if x == 1 else x for x in all_y_true_concatenated]
    all_y_true_concatenated = [
        0 if x == 0 else x for x in all_y_true_concatenated]
    all_y_pred_concatenated = [
        1 if x == 1 else x for x in all_y_pred_concatenated]
    all_y_pred_concatenated = [
        0 if x == 0 else x for x in all_y_pred_concatenated]
    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true_concatenated, all_y_pred_concatenated, average=None)
    return accuracy, baseline, precision, recall, f1, support, macro_f1, all_y_true_concatenated, all_y_pred_concatenated


def evaluate(config, group, name, valid_preds, y_val, test_preds, y_test, y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {}
    if (len(config.valid_ids) > 0): #and (len(config.valid_ids) != len(config.test_ids)):
        results["val_accuracy"] = accuracy_score(y_val, valid_preds)
        cm_val = confusion_matrix(y_val, valid_preds)
        print(classification_report(y_val, valid_preds))
        results["0 count val"] = (y_val == 0).sum()


        #for confusion matrix, get the labels to understand which label it is
        results["val_precision"], results["val_recall"], results["val_fscore"], results["val_support"] = precision_recall_fscore_support(
            y_val, valid_preds, average=None)
        results["val_macroF1"] = f1_score(
            y_val, valid_preds, average="macro")
        print("VAL CM", cm_val)
        results['val_cm'] = cm_val

    # test performance
    if len(config.test_ids) > 0:
        results["test_accuracy"] = accuracy_score(y_test, test_preds)
        cm_test = confusion_matrix(y_test, test_preds)
        results["0 count test"] = (y_test == 0).sum()
        print(classification_report(y_test, test_preds))
        results["test_precision"], results["test_recall"], results["test_fscore"], results["test_support"] = precision_recall_fscore_support(
            y_test, test_preds, average=None)
        results["test_macroF1"] = f1_score(
            y_test, test_preds, average="macro")
        results["test_cm"] = cm_test    
        # wandb.log(results)

    results["y_val"]   = y_val
    results["y_test"]  = y_test
    results["valid_preds"] = valid_preds
    results["test_preds"] = test_preds

    #using get_metrics
    #validation
    print('VALIDATION SET')
    print(len(y_val), len(valid_preds))
    results_val = get_metrics(y_val, valid_preds, tolerance=1)

    #change results_val such that keys have "Val" in the start
    results_val = {f"Val_{k}": v for k, v in results_val.items()}

    print(results_val)
    #testing
    print('TEST SET')
    print(len(y_test), len(test_preds))
    results_test = get_metrics(y_test, test_preds, tolerance=1)
    results_test = {f"Test_{k}": v for k, v in results_test.items()}
    print(results_test)

    return results, results_val, results_test   


def train_miniRocket(config, group, name):
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name=config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling,
                                                                                                                                       config = config)
    
    #if returns none, return none
    if dls is None:
        return None
    # model and train
    model = MiniRocketVotingClassifier(n_estimators=config.n_estimators)
    model.fit(X_train, y_train)

    # evaluate
    if len(config.valid_ids) > 0:
        valid_preds = model.predict(X_val)
    else:
        valid_preds = None
    if len(config.test_ids) > 0:
        test_preds = model.predict(X_test)
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        test_preds_per_participant.append(model.predict(X_p))
    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        val_preds_per_participant.append(model.predict(X_p))

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)
    
    train_metrics = {}

    return model, results, train_metrics, results_val, results_test


def train_fastAI(config, group, name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # special case for final test
    if len(config.valid_ids) == 0:
        config.valid_ids = config.test_ids
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name = config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling,
                                                                                                                                       config = config)
    #if returns none, return none
    if dls is None:
        return None
    # model and train
    cbs = None
    learn = load_config_model(config=config, dls=dls, cbs=cbs)

    print(learn.summary())
    learn.fit_one_cycle(config.n_epoch, config.lr)
    print('RECORDER')
    #get training metrics from training
    train_losses = np.array(learn.recorder.values)[:,0]
    valid_losses = np.array(learn.recorder.values)[:,1]
    #train_accuracies = cb_metrics.train_accuracies
    valid_accuracies = np.array(learn.recorder.values)[:,2]
    #train_f1s = cb_metrics.train_f1s
    valid_f1s = np.array(learn.recorder.values)[:,3]
    #train_precisions = cb_metrics.train_precisions
    valid_precisions = np.array(learn.recorder.values)[:,4]
    #train_recalls = cb_metrics.train_recalls
    valid_recalls = np.array(learn.recorder.values)[:,5]
    
    #save these as a dictionary
    train_metrics = {'train_losses': train_losses, 'valid_losses': valid_losses, 'valid_accuracies': valid_accuracies,  
                     'valid_f1s': valid_f1s, 'valid_precisions': valid_precisions,  'valid_recalls': valid_recalls}

    


    if config.merged_labels:
        majority_class = tensor(1)
    else:
        majority_class = tensor(1)


    # evaluate
    if len(config.valid_ids) > 0:# and len(config.valid_ids) != len(config.test_ids):
        valid_probas, valid_targets= learn.get_X_preds(
            X_val, y_val, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
        #print('VALIDATION SET')
        #print(valid_probas)
        #print(valid_targets)
        #print(valid_preds0)
        valid_preds = [learn.dls.vocab[p]
                       for p in np.argmax(valid_probas, axis=1)]
        print('Number of labels in valid_preds:', Counter(valid_preds))



    else:
        valid_preds = None

    if len(config.test_ids) > 0:
        test_probas, test_targets = learn.get_X_preds(
            X_test, y_test, with_decoded=False)
        test_preds = [learn.dls.vocab[p]
                      for p in np.argmax(test_probas, axis=1)]
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        test_preds_per_participant.append(pred_X_p)

    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        val_preds_per_participant.append(pred_X_p)

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)

    return learn, results, train_metrics, results_val, results_test


def cross_validate(val_fold_size, config, group, name):

    import os
    os.environ["WANDB__SERVICE_WAIT"]="300"

    try:
        if 20 % val_fold_size != 0:
            raise ValueError("val_fold_size must be a divisor of 20")

        

        f1_scores = []
        vsBaselines = []
        f1_scores_accumulated = []
        vsBaselines_accumulated = []

        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        results_all = []

        #if config.dataset_processing exists
        if hasattr(config, 'dataset_processing'):

            dataset_processing = config.dataset_processing
            

            if (dataset_processing == "clean"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05.csv")
                #     df_name = "all_data_05.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_sign.csv")
                #     df_name = "all_data_05_sign.csv"
                df = pd.read_csv("all_participants_0_3.csv")
                df_name = "all_participants_0_3.csv"
                
            elif (dataset_processing == "norm") or (dataset_processing == "pca"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05_norm.csv")
                #     df_name = "all_data_05_norm.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_norm_sign.csv")
                #     df_name = "all_data_05_norm_sign.csv"
                df = pd.read_csv("all_participants_0_3_norm.csv")
                df_name = "all_participants_0_3_norm.csv"
            else:
                df = pd.read_csv(config.df_name)
                df_name = config.df_name
            


        
        #features = df.columns[3:]

        #df_name = config.df_name
        #df = pd.read_csv("../../data/"+df_name)
        features = df.columns[4:]
        config.features = features
        config.df_name = df_name

        #transform participant ids to be in ascendent order from 0 to n_p
        participants_uniques = df['participant'].unique()
        dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
        #print(dict_participant)S
        df['participant'] = df['participant'].map(dict_participant)
        #print(df['session'].unique())
        #print('FINISHED CROSS_VALIDATION')


        train_folds, val_folds, test_folds = create_data_splits_ids(df, "binary")
        #print("Train Folds:", train_folds)
        #print("Val Folds:", val_folds)
        #print("Test Folds:", test_folds)

        # iterate over folds
        for i in range(len(train_folds)):
            # Clear GPU memory before each fold
            torch.cuda.empty_cache()
            train_ids = train_folds[i]
            valid_ids = val_folds[i]
            config.train_ids = train_ids
            config.valid_ids = valid_ids
            config.test_ids = test_folds[i]
            print("Train IDs:", train_ids)
            print("Val IDs:", valid_ids)
            print("Test IDs:", config.test_ids)

            if config.model == "MiniRocket":
                model, results, train_metrics, results_val, results_test = train_miniRocket(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            else:
                model, results, train_metrics, results_val, results_test = train_fastAI(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)

            #if model is none, throw error
            if model is None:
                raise ValueError("Model is None")
            #delete the model from memory
            del model
            # gc.collect()
            torch.cuda.empty_cache()   
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(results_val)
            test_metrics_all.append(results_test)
            results_all.append(results)

            
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # , settings=wandb.Settings(start_method="fork")):
        dataset = config.dataset
        with wandb.init(project=f"minirocket_2025_br_irl_binary_v2", config=config, group="summary-"+group, name=now+"_"+group):

            for i in range(len(train_metrics_all)):
                #get metrics all for fold 0
                train_0 = train_metrics_all[i]
                #modify so name starts with 0
                train_0 = {f"{i}_{k}": v for k, v in train_0.items()}
                #log epochs 
                epochs_log = len(train_0[str(i) + '_train_losses'])
                #wandb.log({'epochs': np.arange(epochs_log)})
                for ep in range(epochs_log):
                    #log all metrics for this epoch
                    epoch_metrics = {}
                    epoch_metrics["epoch"] = ep

                    for key in train_0.keys():
                        epoch_metrics[key] = train_0[key][ep]
                    print('epoch metrics')
                    print(epoch_metrics)

                    wandb.log(epoch_metrics)


                val_0 = val_metrics_all[i]  
                val_0 = {f"{i}_{k}": v for k, v in val_0.items()}
                wandb.log(val_0)
                test_0 = test_metrics_all[i]
                test_0 = {f"{i}_{k}": v for k, v in test_0.items()}
                wandb.log(test_0)
                results_0 = results_all[i]
                results_0 = {f"{i}_{k}": v for k, v in results_0.items()}
                wandb.log({f"{i}_val_cm": results_0[f'{i}_val_cm']})
                wandb.log({f"{i}_test_cm": results_0[f'{i}_test_cm']})


            #now, get average metrics for all folds
            #see keys in train_metrics_all[0]
            keys_train = train_metrics_all[0].keys()
            for key in keys_train:
                #get all values for this key
                values = [d[key] for d in train_metrics_all]
                values = np.array(values)
                #get average
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})

            #same for val
            keys_val = val_metrics_all[0].keys()
            for key in keys_val:
                values = [d[key] for d in val_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})

                
            #same for test
            keys_test = test_metrics_all[0].keys()
            for key in keys_test:
                values = [d[key] for d in test_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})

            #log all the attributes in config
            for key in config.keys():
                wandb.log({key: str(config[key])})
    except Exception as e:
        dataset = config.dataset
        print(e)
        with wandb.init(project=f"minirocket_2025_br_irl_binary_v2", config=config, group="error-"+group, name=group):

            wandb.log({"Error": str(e)})
            for key in config.keys():
                wandb.log({key: str(config[key])})







def cross_validate_bestepoch(val_fold_size, config, group, name):

    import os
    os.environ["WANDB__SERVICE_WAIT"]="300"

    try:
        if 20 % val_fold_size != 0:
            raise ValueError("val_fold_size must be a divisor of 20")

        

        f1_scores = []
        vsBaselines = []
        f1_scores_accumulated = []
        vsBaselines_accumulated = []

        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        results_all = []

        #if config.dataset_processing exists
        if hasattr(config, 'dataset_processing'):

            dataset_processing = config.dataset_processing
            

            if (dataset_processing == "clean"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05.csv")
                #     df_name = "all_data_05.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_sign.csv")
                #     df_name = "all_data_05_sign.csv"
                df = pd.read_csv("all_participants_0_3.csv")
                df_name = "all_participants_0_3.csv"
            elif (dataset_processing == "norm") or (dataset_processing == "pca"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05_norm.csv")
                #     df_name = "all_data_05_norm.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_norm_sign.csv")
                #     df_name = "all_data_05_norm_sign.csv"
                df = pd.read_csv("all_participants_0_3_norm.csv")
                df_name = "all_participants_0_3_norm.csv"
            else:
                df = pd.read_csv(config.df_name)
                df_name = config.df_name

        
        #features = df.columns[3:]

        #df_name = config.df_name
        #df = pd.read_csv("../../data/"+df_name)
        features = df.columns[4:]
        config.features = features
        config.df_name = df_name


        #transform participant ids to be in ascendent order from 0 to n_p
        participants_uniques = df['participant'].unique()
        dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
        #print(dict_participant)
        df['participant'] = df['participant'].map(dict_participant)
        #print(df['session'].unique())
        #print('FINISHED CROSS_VALIDATION')


        train_folds, val_folds, test_folds = create_data_splits_ids(df, "binary")
        #print("Train Folds:", train_folds)
        #print("Val Folds:", val_folds)
        #print("Test Folds:", test_folds)

        # iterate over folds
        for i in range(len(train_folds)):
            # Clear GPU memory before each fold
            torch.cuda.empty_cache()
            train_ids = train_folds[i]
            valid_ids = val_folds[i]
            config.train_ids = train_ids
            config.valid_ids = valid_ids
            config.test_ids = test_folds[i]
            print("Train IDs:", train_ids)
            print("Val IDs:", valid_ids)
            print("Test IDs:", config.test_ids)

            if config.model == "MiniRocket":
                model, results, train_metrics, results_val, results_test = train_miniRocket(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            else:
                model, results, train_metrics, results_val, results_test = train_fastAI(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            torch.cuda.empty_cache()        
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(results_val)
            test_metrics_all.append(results_test)
            results_all.append(results)


        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # , settings=wandb.Settings(start_method="fork")):
        dataset = config.dataset
        with wandb.init(project=f"minirocket_2025_bestepoch_br_irl_binary_v2", config=config, group="summary-"+group, name=now+"_"+group):


            avg_valid_acc = []
            avg_test_acc = []
            for i in range(len(train_metrics_all)):
                val_acc = []
                
                #get metrics all for fold 0
                train_0 = train_metrics_all[i]
                #modify so name starts with 0
                train_0 = {f"{i}_{k}": v for k, v in train_0.items()}
                #log epochs 
                epochs_log = len(train_0[str(i) + '_train_losses'])
                #wandb.log({'epochs': np.arange(epochs_log)})
                for ep in range(epochs_log):
                    #log all metrics for this epoch
                    epoch_metrics = {}
                    epoch_metrics["epoch"] = ep

                    for key in train_0.keys():
                        epoch_metrics[key] = train_0[key][ep]
                    val_acc.append(train_0[f'{i}_valid_accuracies'][ep])

                    print('epoch metrics')
                    print(epoch_metrics)

                    wandb.log(epoch_metrics)

                print(val_acc)
                avg_valid_acc.append(val_acc)

                val_0 = val_metrics_all[i]  
                val_0 = {f"{i}_{k}": v for k, v in val_0.items()}
                wandb.log(val_0)
                test_0 = test_metrics_all[i]
                test_0 = {f"{i}_{k}": v for k, v in test_0.items()}
                wandb.log(test_0)
                results_0 = results_all[i]
                results_0 = {f"{i}_{k}": v for k, v in results_0.items()}
                wandb.log({f"{i}_val_cm": results_0[f'{i}_val_cm']})
                wandb.log({f"{i}_test_cm": results_0[f'{i}_test_cm']})


            #now, get average metrics for all folds
            #get average per epoch of valid acc
            avg_valid_acc = np.array(avg_valid_acc)
            print('AVG VALID ACC')
            print(avg_valid_acc)
            avg_valid_acc = np.mean(avg_valid_acc, axis=0)
            #log it as a list, along with the epoch
            for ep in range(len(avg_valid_acc)):
                wandb.log({f"Epoch": ep, f"Avg_Valid_Acc": avg_valid_acc[ep]})

            #wandb.log({f"Avg_Valid_Acc_all": avg_valid_acc})
            print('mean valid acc')
            print(avg_valid_acc)
            #best epoch, given by the one with highest valid acc
        
            best_epochs = np.argsort(avg_valid_acc)[::-1]
            print('BEST EPOCHS')
            print(best_epochs)
            wandb.log({f"Best_Epoch_valid": list(best_epochs)})
            #if the best epoch is below 30, don't log it and log second best, and so on
            for i in range(len(best_epochs)):
                if best_epochs[i] < 30:
                    continue
                else:
                    wandb.log({f"Best_Epoch_Val": best_epochs[i]})
                    break

        



            #now, get average metrics for all folds
            #see keys in train_metrics_all[0]
            keys_train = train_metrics_all[0].keys()
            for key in keys_train:
                #get all values for this key
                values = [d[key] for d in train_metrics_all]
                #get average
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #same for val
            keys_val = val_metrics_all[0].keys()
            for key in keys_val:
                values = [d[key] for d in val_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                std = np.std(np.array(values))
                wandb.log({f"Avg_{key}": avg})
                wandb.log({f"STD_{key}": std})

                
            #same for test
            keys_test = test_metrics_all[0].keys()
            for key in keys_test:
                values = [d[key] for d in test_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #now for std


            #log all the attributes in config
            for key in config.keys():
                wandb.log({key: str(config[key])})


    except Exception as e:
        dataset = config.dataset
        print(e)
        with wandb.init(project=f"minirocket_2025_bestepoch_br_irl_binary_v2", config=config, group="error-"+group, name=group):

            wandb.log({"Error": str(e)})
            for key in config.keys():
                wandb.log({key: str(config[key])})




def cross_validate_bestmodel(val_fold_size, config, group, name):

    import os
    os.environ["WANDB__SERVICE_WAIT"]="300"

    try:
       
        

        f1_scores = []
        vsBaselines = []
        f1_scores_accumulated = []
        vsBaselines_accumulated = []

        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        results_all = []

        #if config.dataset_processing exists
        if hasattr(config, 'dataset_processing'):

            dataset_processing = config.dataset_processing
            

            if (dataset_processing == "clean"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05.csv")
                #     df_name = "all_data_05.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_sign.csv")
                #     df_name = "all_data_05_sign.csv"
                df = pd.read_csv("all_participants_0_3.csv")
                df_name = "all_participants_0_3.csv"
            elif (dataset_processing == "norm") or (dataset_processing == "pca"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05_norm.csv")
                #     df_name = "all_data_05_norm.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_norm_sign.csv")
                #     df_name = "all_data_05_norm_sign.csv"
                df = pd.read_csv("all_participants_0_3_norm.csv")
                df_name = "all_participants_0_3_norm.csv"
            else:
                df = pd.read_csv(config.df_name)
                df_name = config.df_name

        
        #features = df.columns[3:]

        #df_name = config.df_name
        #df = pd.read_csv("../../data/"+df_name)
        features = df.columns[4:]
        config.features = features
        config.df_name = df_name
        #group_name = config.group


        #transform participant ids to be in ascendent order from 0 to n_p
        participants_uniques = df['participant'].unique()
        dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
        #print(dict_participant)
        df['participant'] = df['participant'].map(dict_participant)
        #print(df['session'].unique())
        #print('FINISHED CROSS_VALIDATION')


        train_folds, val_folds, test_folds = create_data_splits_ids(df, "binary")
        #print("Train Folds:", train_folds)
        #print("Val Folds:", val_folds)
        #print("Test Folds:", test_folds)

        # iterate over folds
        for i in range(len(train_folds)):
            torch.cuda.empty_cache()

            train_ids = train_folds[i]
            valid_ids = val_folds[i]
            config.train_ids = train_ids
            config.valid_ids = valid_ids
            config.test_ids = test_folds[i]
            print("Train IDs:", train_ids)
            print("Val IDs:", valid_ids)
            print("Test IDs:", config.test_ids)

            if config.model == "MiniRocket":
                model, results, train_metrics, results_val, results_test = train_miniRocket_save(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            else:
                model, results, train_metrics, results_val, results_test = train_fastAI_save(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            #f1_scores.append(results["val_macroF1"])
            torch.cuda.empty_cache()    
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(results_val)
            test_metrics_all.append(results_test)
            results_all.append(results)
        #print("Standard Deviation vsBaseline", np.std(vsBaselines))

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # , settings=wandb.Settings(start_method="fork")):
        dataset = config.dataset
        with wandb.init(project=f"minirocket_2025_bestmodel_br_irl_binary_v2", config=config, group="summary-"+group, name=now+"_"+group):


            avg_valid_acc = []
            avg_test_acc = []
            for i in range(len(train_metrics_all)):
                val_acc = []
                
                #get metrics all for fold 0
                train_0 = train_metrics_all[i]
                #modify so name starts with 0
                train_0 = {f"{i}_{k}": v for k, v in train_0.items()}
                #log epochs 
                epochs_log = len(train_0[str(i) + '_train_losses'])
                #wandb.log({'epochs': np.arange(epochs_log)})
                for ep in range(epochs_log):
                    #log all metrics for this epoch
                    epoch_metrics = {}
                    epoch_metrics["epoch"] = ep

                    for key in train_0.keys():
                        epoch_metrics[key] = train_0[key][ep]
                    val_acc.append(train_0[f'{i}_valid_accuracies'][ep])

                    print('epoch metrics')
                    print(epoch_metrics)

                    wandb.log(epoch_metrics)

                print(val_acc)
                avg_valid_acc.append(val_acc)

                val_0 = val_metrics_all[i]  
                val_0 = {f"{i}_{k}": v for k, v in val_0.items()}
                wandb.log(val_0)
                test_0 = test_metrics_all[i]
                test_0 = {f"{i}_{k}": v for k, v in test_0.items()}
                wandb.log(test_0)
                results_0 = results_all[i]
                results_0 = {f"{i}_{k}": v for k, v in results_0.items()}
                wandb.log({f"{i}_val_cm": results_0[f'{i}_val_cm']})
                wandb.log({f"{i}_test_cm": results_0[f'{i}_test_cm']})


    

            #now, get average metrics for all folds
            #see keys in train_metrics_all[0]
            keys_train = train_metrics_all[0].keys()
            for key in keys_train:
                #get all values for this key
                values = [d[key] for d in train_metrics_all]
                #get average
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #same for val
            keys_val = val_metrics_all[0].keys()
            for key in keys_val:
                values = [d[key] for d in val_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                std = np.std(np.array(values))
                wandb.log({f"Avg_{key}": avg})
                wandb.log({f"STD_{key}": std})

                
            #same for test
            keys_test = test_metrics_all[0].keys()
            for key in keys_test:
                values = [d[key] for d in test_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #now for std


            #log all the attributes in config
            for key in config.keys():
                wandb.log({key: str(config[key])})

    except Exception as e:
        dataset = config.dataset
        print(e)
        with wandb.init(project=f"minirocket_2025_bestmodel_br_irl_binary_v2", config=config, group="error-"+group, name=group):

            wandb.log({"Error": str(e)})
            for key in config.keys():
                wandb.log({key: str(config[key])})



def train_miniRocket_save(config, group, name):
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name=config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                    
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)


    model = config.model
    dataset = config.dataset
    end_folder = config.end_folder
    model_numbering = config.model_numbering
    # model and train
    model = MiniRocketVotingClassifier(n_estimators=config.n_estimators)
    model.fit(X_train, y_train)

    #save model and all variables
    with open(f"{end_folder}{group}_{model}_{dataset}_pickle.pkl", "wb") as f:
        pickle.dump(model, f)

    #torch save model and weights
    torch.save(model, f"{end_folder}{group}_{model}_{dataset}_{model_numbering}_model.pth")
    #save checkpoint
    torch.save(model.state_dict(), f"{end_folder}{group}_{model}_{dataset}_{model_numbering}_weights.pth")



    # evaluate
    if len(config.valid_ids) > 0:
        valid_preds = model.predict(X_val)
    else:
        valid_preds = None
    if len(config.test_ids) > 0:
        test_preds = model.predict(X_test)
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        test_preds_per_participant.append(model.predict(X_p))
    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        val_preds_per_participant.append(model.predict(X_p))

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)
    
    train_metrics = {}

    return model, results, train_metrics, results_val, results_test

def train_fastAI_save(config, group, name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # special case for final test
    if len(config.valid_ids) == 0:
        config.valid_ids = config.test_ids
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name = config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)

    model = config.model
    dataset = config.dataset
    end_folder = config.end_folder
    model_numbering = config.model_numbering
    
    # model and train
    cbs = None
    learn = load_config_model(config=config, dls=dls, cbs=cbs)
    print(learn.summary())
    learn.fit_one_cycle(config.n_epoch, config.lr)

    learn.save(f"{end_folder}{group}_{model}_{dataset}_{model_numbering}")

    #learn.save_model(f"{end_folder}{group}_{model}_{dataset}_{model_numbering}_model")
    
    learn.export(f"models/{end_folder}{group}_{model}_{dataset}_{model_numbering}_export.pkl")

    
    
    #save model and all variables
    with open(f"models/{end_folder}{group}_{model}_{dataset}_{model_numbering}_pickle.pkl", "wb") as f:
        pickle.dump(learn, f)

    #torch save model and weights
    #torch.save(model, f"{end_folder}{group}_{model}_{dataset}_model.pth")
    #save checkpoint
    #torch.save(model.state_dict(), f"{end_folder}{group}_{model}_{dataset}_weights.pth")


    print('RECORDER')
    #get training metrics from training
    train_losses = np.array(learn.recorder.values)[:,0]
    valid_losses = np.array(learn.recorder.values)[:,1]
    #train_accuracies = cb_metrics.train_accuracies
    valid_accuracies = np.array(learn.recorder.values)[:,2]
    #train_f1s = cb_metrics.train_f1s
    valid_f1s = np.array(learn.recorder.values)[:,3]
    #train_precisions = cb_metrics.train_precisions
    valid_precisions = np.array(learn.recorder.values)[:,4]
    #train_recalls = cb_metrics.train_recalls
    valid_recalls = np.array(learn.recorder.values)[:,5]
    
    #save these as a dictionary
    train_metrics = {'train_losses': train_losses, 'valid_losses': valid_losses, 'valid_accuracies': valid_accuracies,  
                     'valid_f1s': valid_f1s, 'valid_precisions': valid_precisions,  'valid_recalls': valid_recalls}

    


    if config.merged_labels:
        majority_class = tensor(1)
    else:
        majority_class = tensor(1)


    # evaluate
    if len(config.valid_ids) > 0 :#and len(config.valid_ids) != len(config.test_ids):
        valid_probas, valid_targets= learn.get_X_preds(
            X_val, y_val, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
        #print('VALIDATION SET')
        #print(valid_probas)
        #print(valid_targets)
        #print(valid_preds0)
        valid_preds = [learn.dls.vocab[p]
                       for p in np.argmax(valid_probas, axis=1)]
        print('Number of labels in valid_preds:', Counter(valid_preds))



    else:
        valid_preds = None

    if len(config.test_ids) > 0:
        test_probas, test_targets = learn.get_X_preds(
            X_test, y_test, with_decoded=False)
        test_preds = [learn.dls.vocab[p]
                      for p in np.argmax(test_probas, axis=1)]
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        test_preds_per_participant.append(pred_X_p)

    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        val_preds_per_participant.append(pred_X_p)

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)

    return learn, results, train_metrics, results_val, results_test






def train_miniRocket_pretrain(config, group, name):
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name=config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)

    #reload model
    model = config.model
    dataset = config.dataset
    end_folder = config.end_folder
    model_pretrained = config.model_pretrained
    dataset_pretrained = config.dataset_pretrained
    end_folder_pretrained = config.end_folder_pretrained
    model_numbering = config.model_numbering

    # model and train
    model = MiniRocketVotingClassifier(n_estimators=config.n_estimators)
    #load model and weights, using torch.load
    model.load_state_dict(torch.load(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}_weights.pth"))




    model.fit(X_train, y_train)

    #save model and all variables
    with open(f"{end_folder}posttrained_{model}_{dataset}_{model_numbering}_pickle.pkl", "wb") as f:
        pickle.dump(model, f)

    #torch save model and weights
    torch.save(model, f"{end_folder}posttrained_{model}_{dataset}_{model_numbering}_model.pth")
    #save checkpoint
    torch.save(model.state_dict(), f"{end_folder}posttrained_{model}_{dataset}_{model_numbering}_weights.pth")



    # evaluate
    if len(config.valid_ids) > 0:
        valid_preds = model.predict(X_val)
    else:
        valid_preds = None
    if len(config.test_ids) > 0:
        test_preds = model.predict(X_test)
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        test_preds_per_participant.append(model.predict(X_p))
    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        val_preds_per_participant.append(model.predict(X_p))

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)
    
    train_metrics = {}

    return model, results, train_metrics, results_val, results_test


def train_fastAI_pretrain(config, group, name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # special case for final test
    if len(config.valid_ids) == 0:
        config.valid_ids = config.test_ids
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(df_name=config.df_name,
                                                                                                                                       threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)

    #reload model
    model = config.model
    dataset = config.dataset
    end_folder = config.end_folder
    model_pretrained = config.model_pretrained
    dataset_pretrained = config.dataset_pretrained
    end_folder_pretrained = config.end_folder_pretrained
    model_numbering = config.model_numbering

    # model and train
    cbs = None
    learn = load_config_model_final(config=config, dls=dls, cbs=cbs)
     #load model and weights, using torch.load

    #learn.freeze()
    learn.summary()
    # Unfreeze the last layer
    for name, param in learn.model.named_parameters():
        print(name) 
        if model == 'LSTM_FCN' or model == 'GRU_FCN':
            if 'fc' in name:  # Assuming the last layer is a fully connected layer named 'fc'
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif model == 'gMLP':
            #last layer is a linear layer
            if 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif model == 'InceptionTimePlus':
            #last layer is a linear layer
            if 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False





    print(learn.summary())



    learn.fit_one_cycle(config.n_epoch, config.lr)

    #learn.save_all(f"{end_folder}pretrain_{model}_{dataset}_{model_numbering}", verbose=True)

    learn.save(f"{end_folder}{group}_{model}_{dataset}_{model_numbering}")

    #learn.save_model(f"{end_folder}{group}_{model}_{dataset}_{model_numbering}_model")
    
    learn.export(f"models/{end_folder}{group}_{model}_{dataset}_{model_numbering}_export.pkl")

    
    
    
    #save model and all variables
    with open(f"models/{end_folder}pretrain_{model}_{dataset}_{model_numbering}_pickle.pkl", "wb") as f:
        pickle.dump(learn, f)


    print('RECORDER')

    #print(learn.recorder.values)
    #get training metrics from training
    train_losses = np.array(learn.recorder.values)[:,0]
    valid_losses = np.array(learn.recorder.values)[:,1]
    #train_accuracies = cb_metrics.train_accuracies
    valid_accuracies = np.array(learn.recorder.values)[:,2]
    #train_f1s = cb_metrics.train_f1s
    valid_f1s = np.array(learn.recorder.values)[:,3]
    #train_precisions = cb_metrics.train_precisions
    valid_precisions = np.array(learn.recorder.values)[:,4]
    #train_recalls = cb_metrics.train_recalls
    valid_recalls = np.array(learn.recorder.values)[:,5]
    
    #save these as a dictionary
    train_metrics = {'train_losses': train_losses, 'valid_losses': valid_losses, 'valid_accuracies': valid_accuracies,  
                     'valid_f1s': valid_f1s, 'valid_precisions': valid_precisions,  'valid_recalls': valid_recalls}

    


    if config.merged_labels:
        majority_class = tensor(1)
    else:
        majority_class = tensor(1)


    # evaluate
    print('evaluating')
    if len(config.valid_ids) > 0:# and len(config.valid_ids) != len(config.test_ids):
        print(X_val.shape, y_val.shape)
        valid_probas, valid_targets= learn.get_X_preds(
            X_val, y_val, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
        print('VALIDATION SET')

        #print(valid_probas)
        #
        # print(valid_targets)
        #print(valid_preds0)
        valid_preds = [learn.dls.vocab[p]
                       for p in np.argmax(valid_probas, axis=1)]
        print('Number of labels in valid_preds:', Counter(valid_preds))



    else:
        valid_preds = None

    if len(config.test_ids) > 0:
        test_probas, test_targets = learn.get_X_preds(
            X_test, y_test, with_decoded=False)
        test_preds = [learn.dls.vocab[p]
                      for p in np.argmax(test_probas, axis=1)]
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        test_preds_per_participant.append(pred_X_p)

    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        val_preds_per_participant.append(pred_X_p)

    results, results_val, results_test = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)

    return learn, results, train_metrics, results_val, results_test



def load_config_model_final(config, dls, cbs):
    end_folder = config.end_folder
    model_pretrained = config.model_pretrained
    dataset_pretrained = config.dataset_pretrained
    end_folder_pretrained = config.end_folder_pretrained
    model_numbering = config.model_numbering

    if config.merged_labels:  # baseline metric has hardcoded majority class and needs to know if labels are merged
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro")]
    else:
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro")]
        
    cb_metrics = MetricsCallback()


    if config.model == "InceptionTime":
        model = InceptionTime(dls.vars, dls.c)
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn

    elif config.model == "InceptionTimePlus":
        #return learn
        learn = load_learner(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}_export.pkl")
        print('got learn')
        old_model = learn.model
        print(old_model)


        new_in_features = dls.vars # Get the number of input features from your DataLoaders
        print('new_in_features', new_in_features)
        out_features = 2  # Keep the same number of output classes
        modified_model = modify_inception_model(old_model, new_in_features, out_features)
        print('finalized model')
        model = modified_model
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        print('finished learner')
        return learn

    elif config.model == "TST":
        model = TST(dls.vars, dls.c, dls.len,
                    dropout=config.dropout_TST, fc_dropout=config.fc_dropout_TST) 
        if config.focal_loss:
           learn = Learner(dls, model, metrics=metrics,cbs = cbs, loss_func=FocalLossFlat()) 
        else:
            learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(),  metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn

    elif config.model == "XceptionTime":
        model = XceptionTime(dls.vars, dls.c)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn

    elif config.model == "ResNet":
        model = ResNet(dls.vars, dls.c)
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "xresnet1d34":
        model = xresnet1d34(dls.vars, dls.c)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "ResCNN":
        model = ResCNN(dls.vars, dls.c)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "OmniScaleCNN":
        model = OmniScaleCNN(dls.vars, dls.c, dls.len)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "mWDN":
        model = mWDN(dls.vars, dls.c, dls.len)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "LSTM_FCN":
        #return learn
        
        learn = load_learner(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}_export.pkl")
        print('got learn')
        old_model = learn.model
        print('old model')
        print(old_model)

        new_in_features = dls.vars # Get the number of input features from your DataLoaders
        print('new_in_features', new_in_features)
        out_features = 2  # Keep the same number of output classes
        modified_model = modify_lstm_model(old_model, new_in_features, out_features)
        print('finalized model')
        model = modified_model
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        print('finished learner')
        return learn
    elif config.model == "GRU_FCN":
        #return learn
        learn = load_learner(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}_export.pkl")
        print('got learn')
        old_model = learn.model
        print('old model')
        print(old_model)

        new_in_features = dls.vars # Get the number of input features from your DataLoaders
        print('new_in_features', new_in_features)
        out_features = 2  # Keep the same number of output classes
        modified_model = modify_lstm_model(old_model, new_in_features, out_features)
        print('finalized model')
        model = modified_model
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        print('finished learner')
        return learn

    elif config.model == "LSTM":
        model = LSTM(dls.vars, dls.c, n_layers=3, bidirectional=True)

        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        learn = load_all(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}", verbose=True)
        return learn
    elif config.model == "gMLP":
        
        learn = load_learner(f"{end_folder_pretrained}best_{model_pretrained}_{dataset_pretrained}_{model_numbering}_export.pkl")
        print('got learn')
        #get description of learn
        #print(learn.model)
        old_model = learn.model

        new_in_features = dls.vars # Get the number of input features from your DataLoaders
        print('new_in_features', new_in_features)
        out_features = 2  # Keep the same number of output classes
        modified_model = modify_gmlp_model(old_model, new_in_features, out_features)

        print('finalized model')
        
        #model = gMLP(dls.vars, dls.c, dls.len)
        #learn = Learner(dls, model, metrics=metrics, cbs=cbs)
        model = modified_model
        if config.focal_loss:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss()) 
        else:
            learn = Learner(dls, model, metrics=metrics, cbs=cbs) 
        print('finished learner')
        return learn





def modify_gmlp_model(model, new_in_features, out_features=2):
    # Modify the first layer to accept new input features
    old_first_layer = model.patcher
    #add conv1d layer
    


    new_first_layer = torch.nn.Conv1d(new_in_features, old_first_layer.out_channels, kernel_size=(1,), stride=(1,))
    
    print('here')
    print(new_first_layer.weight)
    # Initialize the weights of the new first layer
    nn.init.xavier_uniform_(new_first_layer.weight)
    nn.init.zeros_(new_first_layer.bias)
    
    # Replace the first layer in the model
    model.patcher = new_first_layer

    # Modify the last layer to ensure correct output features
    old_last_layer = model.head
    new_last_layer = nn.Linear(old_last_layer.in_features, out_features)
    
    # Initialize the weights of the new last layer
    nn.init.xavier_uniform_(new_last_layer.weight)
    nn.init.zeros_(new_last_layer.bias)
    
    # Replace the last layer in the model
    model.head = new_last_layer

    return model
    
def modify_lstm_model(model, new_in_features, out_features=2):
    # Modify the first layer to accept new input features
    old_first_layer = model.convblock1
    conv_layers = old_first_layer[0].out_channels
    #add conv1d layer
    


    new_first_layer = torch.nn.Conv1d(new_in_features, conv_layers, kernel_size=(7,), stride=(1,))
    
    #print('here')
    #print(new_first_layer.weight)
    # Initialize the weights of the new first layer
    nn.init.xavier_uniform_(new_first_layer.weight)
    nn.init.zeros_(new_first_layer.bias)
    
    # Replace the first layer in the model
    model.convblock1  = new_first_layer

    # Modify the last layer to ensure correct output features
    old_last_layer = model.fc
    new_last_layer = nn.Linear(old_last_layer.in_features, out_features)
    
    # Initialize the weights of the new last layer
    nn.init.xavier_uniform_(new_last_layer.weight)
    nn.init.zeros_(new_last_layer.bias)
    
    # Replace the last layer in the model
    model.fc = new_last_layer

    #unfreeze the model
    #model.unfreeze()
    #print('model unfrozen')
    print(model)

    return model


def modify_inception_model(model, new_in_features, out_features=2):
    # Modify the first layer to accept new input features
    first_conv_layers = model.backbone[0].inception[0].bottleneck[0].out_channels
    
    # Add conv1d layer
    new_first_layer = torch.nn.Conv1d(new_in_features, first_conv_layers, kernel_size=(1,), stride=(1,), bias=False)
    
    # Initialize the weights of the new first layer
    nn.init.xavier_uniform_(new_first_layer.weight)
    if new_first_layer.bias is not None:
        nn.init.zeros_(new_first_layer.bias)
    
    # Replace the first layer in the model
    model.backbone[0].inception[0].bottleneck[0] = new_first_layer

    # Modify the Conv1d layer inside the mp_conv block
    mp_conv_layers = model.backbone[0].inception[0].mp_conv[1][0].out_channels
    new_mp_conv_layer = torch.nn.Conv1d(new_in_features, mp_conv_layers, kernel_size=(1,), stride=(1,), bias=False)
    
    # Initialize the weights of the new mp_conv layer
    nn.init.xavier_uniform_(new_mp_conv_layer.weight)
    if new_mp_conv_layer.bias is not None:
        nn.init.zeros_(new_mp_conv_layer.bias)
    
    # Replace the Conv1d layer inside the mp_conv block
    model.backbone[0].inception[0].mp_conv[1][0] = new_mp_conv_layer

    # Modify the Conv1d layer inside the shortcut block
    shortcut_conv_layers = model.backbone[0].shortcut[0][0].out_channels
    new_shortcut_layer = torch.nn.Conv1d(new_in_features, shortcut_conv_layers, kernel_size=(1,), stride=(1,), bias=False)
    
    # Initialize the weights of the new shortcut layer
    nn.init.xavier_uniform_(new_shortcut_layer.weight)
    if new_shortcut_layer.bias is not None:
        nn.init.zeros_(new_shortcut_layer.bias)
    
    # Replace the Conv1d layer inside the shortcut block
    model.backbone[0].shortcut[0][0] = new_shortcut_layer

    # Modify the last layer to ensure correct output features
    old_last_layer = model.head[0][1][0]
    new_last_layer = nn.Linear(old_last_layer.in_features, out_features)
    
    # Initialize the weights of the new last layer
    nn.init.xavier_uniform_(new_last_layer.weight)
    if new_last_layer.bias is not None:
        nn.init.zeros_(new_last_layer.bias)
    
    # Replace the last layer in the model
    model.head[0][1][0] = new_last_layer
    
    return model



def cross_validate_pretrained(val_fold_size, config, group, name):

    import os
    os.environ["WANDB__SERVICE_WAIT"]="300"

    try:
       
        

        f1_scores = []
        vsBaselines = []
        f1_scores_accumulated = []
        vsBaselines_accumulated = []

        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        results_all = []

        #if config.dataset_processing exists
        if hasattr(config, 'dataset_processing'):

            dataset_processing = config.dataset_processing
            

            if (dataset_processing == "clean"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05.csv")
                #     df_name = "all_data_05.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_sign.csv")
                #     df_name = "all_data_05_sign.csv"
                df = pd.read_csv("all_participants_0_3.csv")
                df_name = "all_participants_0_3.csv"
            elif (dataset_processing == "norm") or (dataset_processing == "pca"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05_norm.csv")
                #     df_name = "all_data_05_norm.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_norm_sign.csv")
                #     df_name = "all_data_05_norm_sign.csv"
                df = pd.read_csv("all_participants_0_3_norm.csv")
                df_name = "all_participants_0_3_norm.csv"
            else:
                df = pd.read_csv(config.df_name)
                df_name = config.df_name

        
        #features = df.columns[3:]

        #df_name = config.df_name
        #df = pd.read_csv("../../data/"+df_name)
        features = df.columns[4:]
        config.features = features
        config.df_name = df_name


        #transform participant ids to be in ascendent order from 0 to n_p
        participants_uniques = df['participant'].unique()
        dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
        #print(dict_participant)
        df['participant'] = df['participant'].map(dict_participant)
        #print(df['session'].unique())
        #print('FINISHED CROSS_VALIDATION')


        train_folds, val_folds, test_folds = create_data_splits_ids(df, "binary")
        #print("Train Folds:", train_folds)
        #print("Val Folds:", val_folds)
        #print("Test Folds:", test_folds)

        # iterate over folds
        for i in range(len(train_folds)):
            # Clear GPU memory before each fold
            torch.cuda.empty_cache()

            train_ids = train_folds[i]
            valid_ids = val_folds[i]
            config.train_ids = train_ids
            config.valid_ids = valid_ids
            config.test_ids = test_folds[i]
            print("Train IDs:", train_ids)
            print("Val IDs:", valid_ids)
            print("Test IDs:", config.test_ids)

            if config.model == "MiniRocket":
                model, results, train_metrics, results_val, results_test = train_miniRocket_pretrain(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            else:
                model, results, train_metrics, results_val, results_test = train_fastAI_pretrain(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            torch.cuda.empty_cache()
            #f1_scores.append(results["val_macroF1"])
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(results_val)
            test_metrics_all.append(results_test)
            results_all.append(results)
        #print("Standard Deviation vsBaseline", np.std(vsBaselines))

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # , settings=wandb.Settings(start_method="fork")):
        dataset = config.dataset
        with wandb.init(project=f"minirocket_2025_br_irl_binary_v2", config=config, group="summary-"+group, name=now+"_"+group):


            avg_valid_acc = []
            avg_test_acc = []
            for i in range(len(train_metrics_all)):
                val_acc = []
                
                #get metrics all for fold 0
                train_0 = train_metrics_all[i]
                #modify so name starts with 0
                train_0 = {f"{i}_{k}": v for k, v in train_0.items()}
                #log epochs 
                epochs_log = len(train_0[str(i) + '_train_losses'])
                #wandb.log({'epochs': np.arange(epochs_log)})
                for ep in range(epochs_log):
                    #log all metrics for this epoch
                    epoch_metrics = {}
                    epoch_metrics["epoch"] = ep

                    for key in train_0.keys():
                        epoch_metrics[key] = train_0[key][ep]
                    val_acc.append(train_0[f'{i}_valid_accuracies'][ep])

                    print('epoch metrics')
                    print(epoch_metrics)

                    wandb.log(epoch_metrics)

                print(val_acc)
                avg_valid_acc.append(val_acc)

                val_0 = val_metrics_all[i]  
                val_0 = {f"{i}_{k}": v for k, v in val_0.items()}
                wandb.log(val_0)
                test_0 = test_metrics_all[i]
                test_0 = {f"{i}_{k}": v for k, v in test_0.items()}
                wandb.log(test_0)
                results_0 = results_all[i]
                results_0 = {f"{i}_{k}": v for k, v in results_0.items()}
                wandb.log({f"{i}_val_cm": results_0[f'{i}_val_cm']})
                wandb.log({f"{i}_test_cm": results_0[f'{i}_test_cm']})


            #now, get average metrics for all folds
            #get average per epoch of valid acc
            avg_valid_acc = np.array(avg_valid_acc)
            print('AVG VALID ACC')
            print(avg_valid_acc)
            avg_valid_acc = np.mean(avg_valid_acc, axis=0)
            #log it as a list, along with the epoch
            for ep in range(len(avg_valid_acc)):
                wandb.log({f"Epoch": ep, f"Avg_Valid_Acc": avg_valid_acc[ep]})

            #wandb.log({f"Avg_Valid_Acc_all": avg_valid_acc})
            print('mean valid acc')
            print(avg_valid_acc)
            #best epoch, given by the one with highest valid acc
        
            best_epochs = np.argsort(avg_valid_acc)[::-1]
            print('BEST EPOCHS')
            print(best_epochs)
            wandb.log({f"Best_Epoch_valid": list(best_epochs)})
            #if the best epoch is below 30, don't log it and log second best, and so on
            for i in range(len(best_epochs)):
                if best_epochs[i] < 30:
                    continue
                else:
                    wandb.log({f"Best_Epoch_Val": best_epochs[i]})
                    break



    

            #now, get average metrics for all folds
            #see keys in train_metrics_all[0]
            keys_train = train_metrics_all[0].keys()
            for key in keys_train:
                #get all values for this key
                values = [d[key] for d in train_metrics_all]
                #get average
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #same for val
            keys_val = val_metrics_all[0].keys()
            for key in keys_val:
                values = [d[key] for d in val_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                std = np.std(np.array(values))
                wandb.log({f"Avg_{key}": avg})
                wandb.log({f"STD_{key}": std})

                
            #same for test
            keys_test = test_metrics_all[0].keys()
            for key in keys_test:
                values = [d[key] for d in test_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_{key}": std})

            #now for std


            #log all the attributes in config
            for key in config.keys():
                wandb.log({key: str(config[key])})

    except Exception as e:
        dataset = config.dataset
        print(e)
        with wandb.init(project=f"minirocket_2025_br_irl_binary_v2", config=config, group="error-"+group, name=group):

            wandb.log({"Error": str(e)})
            for key in config.keys():
                wandb.log({key: str(config[key])})


def cross_validate_pp(val_fold_size, config, group, name):

    import os
    os.environ["WANDB__SERVICE_WAIT"]="300"

    try:
        if 20 % val_fold_size != 0:
            raise ValueError("val_fold_size must be a divisor of 20")

        

        f1_scores = []
        vsBaselines = []
        f1_scores_accumulated = []
        vsBaselines_accumulated = []

        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        results_all = []

        #if config.dataset_processing exists
        if hasattr(config, 'dataset_processing'):

            dataset_processing = config.dataset_processing
            

            if (dataset_processing == "clean"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05.csv")
                #     df_name = "all_data_05.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_sign.csv")
                #     df_name = "all_data_05_sign.csv"

                df = pd.read_csv("all_participants_0_3.csv")
                df_name = "all_participants_0_3.csv"
            elif (dataset_processing == "norm") or (dataset_processing == "pca"):
                # if config.groundtruth == 'multi':
                #     df = pd.read_csv("../../data/all_data_05_norm.csv")
                #     df_name = "all_data_05_norm.csv"
                # else:
                #     df = pd.read_csv("../../data/all_data_05_norm_sign.csv")
                #     df_name = "all_data_05_norm_sign.csv"
                df = pd.read_csv("all_participants_0_3_norm.csv")
                df_name = "all_participants_0_3_norm.csv"
            else:
                df = pd.read_csv(config.df_name)
                df_name = config.df_name



        
        #features = df.columns[3:]

        #df_name = config.df_name
        #df = pd.read_csv("../../data/"+df_name)
        features = df.columns[4:]
        config.features = features
        config.df_name = df_name

        #transform participant ids to be in ascendent order from 0 to n_p
        participants_uniques = df['participant'].unique()
        dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
        #print(dict_participant)S
        df['participant'] = df['participant'].map(dict_participant)
        #print(df['session'].unique())



        # iterate over folds
        for i in range(len(participants_uniques)):

            if config.model == "MiniRocket":
                model, results, train_metrics, results_val, results_test = train_miniRocket(
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)
            else:
                model, results, train_metrics, results_val, results_test = train_fastAI_pp(participant = i,
                    config=config, group=group, name="_iteration"+str(i)+"_var"+name)

            torch.cuda.empty_cache()

            train_metrics_all.append(train_metrics)
            val_metrics_all.append(results_val)
            test_metrics_all.append(results_test)
            results_all.append(results)



        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # , settings=wandb.Settings(start_method="fork")):
        dataset = config.dataset
        with wandb.init(project=f"minirocket_2025_pp_br_irl_binary_v2", config=config, group="summary-"+group, name=now+"_"+group):

            avg_valid_acc = []
            avg_test_acc = []
            for i in range(len(train_metrics_all)):
                val_acc = []
                
                #get metrics all for fold 0
                train_0 = train_metrics_all[i]
                #modify so name starts with 0
                train_0 = {f"{i}_{k}": v for k, v in train_0.items()}
                #log epochs 
                epochs_log = len(train_0[str(i) + '_train_losses'])
                #wandb.log({'epochs': np.arange(epochs_log)})
                for ep in range(epochs_log):
                    #log all metrics for this epoch
                    epoch_metrics = {}
                    epoch_metrics["epoch"] = ep

                    for key in train_0.keys():
                        epoch_metrics[key] = train_0[key][ep]
                    val_acc.append(train_0[f'{i}_valid_accuracies'][ep])

                    print('epoch metrics')
                    print(epoch_metrics)

                    wandb.log(epoch_metrics)

                print(val_acc)
                avg_valid_acc.append(val_acc)

                val_0 = val_metrics_all[i]  
                val_0 = {f"{i}_{k}": v for k, v in val_0.items()}
                wandb.log(val_0)
                test_0 = test_metrics_all[i]
                test_0 = {f"{i}_{k}": v for k, v in test_0.items()}
                wandb.log(test_0)
                results_0 = results_all[i]
                results_0 = {f"{i}_{k}": v for k, v in results_0.items()}
                wandb.log({f"{i}_val_cm": results_0[f'{i}_val_cm']})
                wandb.log({f"{i}_test_cm": results_0[f'{i}_test_cm']})


            #now, get average metrics for all folds
            #get average per epoch of valid acc
            avg_valid_acc = np.array(avg_valid_acc)
            print('AVG VALID ACC')
            print(avg_valid_acc)
            avg_valid_acc = np.mean(avg_valid_acc, axis=0)
            #log it as a list, along with the epoch
            for ep in range(len(avg_valid_acc)):
                wandb.log({f"Epoch": ep, f"Avg_Valid_Acc": avg_valid_acc[ep]})

            #wandb.log({f"Avg_Valid_Acc_all": avg_valid_acc})
            print('mean valid acc')
            print(avg_valid_acc)
            #best epoch, given by the one with highest valid acc
        
            best_epochs = np.argsort(avg_valid_acc)[::-1]
            print('BEST EPOCHS')
            print(best_epochs)
            wandb.log({f"Best_Epoch_valid": list(best_epochs)})
            #if the best epoch is below 30, don't log it and log second best, and so on
            for i in range(len(best_epochs)):
                if best_epochs[i] < 30:
                    continue
                else:
                    wandb.log({f"Best_Epoch_Val": best_epochs[i]})
                    break

        
            #now, get average metrics for all folds
            #see keys in train_metrics_all[0]
            keys_train = train_metrics_all[0].keys()
            for key in keys_train:
                #get all values for this key
                values = [d[key] for d in train_metrics_all]
                values = np.array(values)
                #get average
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_TRAIN_{key}": avg})

            #same for val
            keys_val = val_metrics_all[0].keys()
            for key in keys_val:
                values = [d[key] for d in val_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_Val_{key}": avg})

                
            #same for test
            keys_test = test_metrics_all[0].keys()
            for key in keys_test:
                values = [d[key] for d in test_metrics_all]
                values = np.array(values)
                avg = np.mean(values, axis=0)
                wandb.log({f"Avg_Test_{key}": avg})
                std = np.std(np.array(values))
                wandb.log({f"STD_Test_{key}": std})

            #log all the attributes in config
            for key in config.keys():
                wandb.log({key: str(config[key])})

    except Exception as e:
        dataset = config.dataset
        print(e)
        with wandb.init(project=f"minirocket_2025_pp_br_irl_binary_v2", config=config, group="error-"+group, name=group):

            wandb.log({"Error": str(e)})
            for key in config.keys():
                wandb.log({key: str(config[key])})



def train_fastAI_pp(participant, config, group, name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # special case for final test
    #if len(config.valid_ids) == 0:
    #    config.valid_ids = config.test_ids
    # prepare data
    print('now')
    dls, X_train, y_train, X_val, y_val, X_test, y_test = dataPrep_pp(df_name = config.df_name,
                                                                    participant = participant,
                                                                    participant_minimum=config.participant_minimum,
                                                                    threshold=config.threshold,
                                                                    interval_length=config.interval_length,
                                                                    stride_train=config.stride_train,
                                                                    stride_eval=config.stride_eval,
                                                                    use_lvl1=config.use_lvl1,
                                                                    use_lvl2=config.use_lvl2,
                                                                    merge_labels=config.merged_labels,
                                                                    batch_size=config.batch_size,
                                                                    batch_tfms=config.batch_tfms,
                                                                    features=config.features,
                                                                    verbose=config.verbose,
                                                                    context_length=config.context_length,
                                                                    oversampling=config.oversampling,
                                                                    undersampling=config.undersampling,
                                                                    config=config)

    # model and train
    cbs = None
    learn = load_config_model(config=config, dls=dls, cbs=cbs)
    learn.fit_one_cycle(config.n_epoch, config.lr)
    print('RECORDER')
    #get training metrics from training
    train_losses = np.array(learn.recorder.values)[:,0]
    valid_losses = np.array(learn.recorder.values)[:,1]
    #train_accuracies = cb_metrics.train_accuracies
    valid_accuracies = np.array(learn.recorder.values)[:,2]
    #train_f1s = cb_metrics.train_f1s
    valid_f1s = np.array(learn.recorder.values)[:,3]
    #train_precisions = cb_metrics.train_precisions
    valid_precisions = np.array(learn.recorder.values)[:,4]
    #train_recalls = cb_metrics.train_recalls
    valid_recalls = np.array(learn.recorder.values)[:,5]
    
    #save these as a dictionary
    train_metrics = {'train_losses': train_losses, 'valid_losses': valid_losses, 'valid_accuracies': valid_accuracies,  
                     'valid_f1s': valid_f1s, 'valid_precisions': valid_precisions,  'valid_recalls': valid_recalls}

    


    if config.merged_labels:
        majority_class = tensor(1)
    else:
        majority_class = tensor(1)


    # evaluate
    valid_probas, valid_targets= learn.get_X_preds(
        X_val, y_val, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
    #print('VALIDATION SET')
    #print(valid_probas)
    #print(valid_targets)
    #print(valid_preds0)
    valid_preds = [learn.dls.vocab[p]
                    for p in np.argmax(valid_probas, axis=1)]
    print('Number of labels in valid_preds:', Counter(valid_preds))


    test_probas, test_targets = learn.get_X_preds(
            X_test, y_test, with_decoded=False)
    test_preds = [learn.dls.vocab[p]
                      for p in np.argmax(test_probas, axis=1)]
    
    results, results_val, results_test = evaluate_pp(config, group, name, valid_preds, y_val, test_preds, y_test)

    return learn, results, train_metrics, results_val, results_test



def dataPrep_pp(df_name, participant, participant_minimum, threshold, interval_length, stride_train, stride_eval, use_lvl1, use_lvl2 , merge_labels, batch_size, batch_tfms, features, context_length, oversampling, undersampling,config=config, verbose=True):
    '''
    :param threshold: threshold for AoI
    :param interval_length: how many frames make up one datapoint
    :param stride_train: how many frames are skipped between two datapoints (train)
    :param stride_eval: how many frames are skipped between two datapoints (eval)
    :param train_ids: list of participant ids for training
    :param valid_ids: list of participant ids for validation
    :param test_ids: list of participant ids for testing
    :param use_lvl1: boolean indicating if level 1 data is used
    :param use_lvl2: boolean indicating if level 2 data is used
    :param merge_labels: boolean indicating if confusion and error are merged
    :param batch_size: batch size for training
    :param batch_tfms: transformations
    :param features: list of timeseries features used
    :param context_length: length of context
    :param oversampling: boolean indicating if oversampling is used
    :param undersampling: boolean indicating if undersampling is used

    :return: dataloader and individual datasets/labels
    '''

    #print('CHECKING UPON DATA PREP')
    #print('valid_ids', valid_ids)
    #print('test_ids', test_ids)
    lvl1, data = read_in_data_pp(df_name = df_name, participant = participant, participant_minimum = participant_minimum, threshold=threshold, interval_length=interval_length,
                                    stride_train=stride_train, stride_eval=stride_eval, features=features, context_length=context_length, config=config)


    #print('lvl1', len(lvl1))
    # prepare labels (1d array) and data (3D array) for TSAI
    features = data['train'].columns[4:]
    print('FEATURES')
    print(features)
    X_train = np.empty(
        (0, len(features), interval_length+context_length), dtype=np.float64)
    X_val = np.empty((0, len(features),
                     interval_length+context_length), dtype=np.float64)
    X_test = np.empty((0, len(features),
                      interval_length+context_length), dtype=np.float64)
    y_train = np.empty(0)
    y_val = np.empty(0)
    y_test = np.empty(0)

    # for final eval after concatenation
    y_test_raw = []
    y_val_raw = []
    X_val_by_participant = []
    X_test_by_participant = []

    #print('use lvl1', use_lvl1)
    #print('train_ids', train_ids)
    #print('valid_ids', valid_ids)
    #print('test_ids', test_ids)
    #print(len(lvl1))

    if use_lvl1:
        y_train = np.append(y_train, lvl1[0][1])
        #print(np.array(lvl1[i][0]).shape)
        #print(np.array(lvl1[i][1]).shape)

        X_train = np.append(X_train, lvl1[0][0], axis=0)
    if use_lvl1:
        y_val = np.append(y_val, lvl1[1][1])
        X_val = np.append(X_val, lvl1[1][0], axis=0)
        
    if use_lvl1:
            y_test = np.append(y_test, lvl1[2][1])
            X_test = np.append(X_test, lvl1[2][0], axis=0)
            
        

    if oversampling:
        ros = RandomOverSampler(random_state=0, sampling_strategy={0: Counter(y_train)[
                                0], 1: Counter(y_train)[1]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]
    if undersampling:
        ros = RandomUnderSampler(random_state=0, sampling_strategy={"Normal": Counter(y_train)[
                                 "Confusion"], "Confusion": Counter(y_train)["Confusion"], "Error": Counter(y_train)["Error"]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]

    if merge_labels:
        y_train[y_train == "Error"] = "Confusion"
        if len(y_val) > 0:
            y_val[y_val == "Error"] = "Confusion"
        if len(y_test) > 0:
            y_test[y_test == "Error"] = "Confusion"

    if verbose:
        print("Train Labels:", y_train.shape)
        print("Val Labels", y_val.shape)
        print("Test Labels:", y_test.shape)
        print("Train Data:", X_train.shape)
        print("Val Data:", X_val.shape)
        print("Test Data:", X_test.shape)
        #count of lables
        print("Train Labels:", Counter(y_train))
        print("Val Labels:", Counter(y_val))
        print("Test Labels:", Counter(y_test))

    X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])

    tfms = [None, TSClassification()]  # transforms for the data
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[
                                   batch_size, 128], batch_tfms=batch_tfms, num_workers=0)
    print('DATALOADERS')
    print(dls.c)
    return dls, X_train, y_train, X_val, y_val, X_test, y_test

def read_in_data_pp(df_name, participant, participant_minimum, threshold, interval_length, stride_train, stride_eval, features, context_length, config):
    """
    reads in merged csvs: applies recompute_treshold() and create_single_label_per_interval

    :param threshold: threshold for AoI analysis
    :param interval_length: length of one of the resulting samples
    :param stride_train: by how many frames the next interval sample is moved. Non-overlaping if stride==interval_length
    :param stride_eval: by how many frames the next interval sample is moved (used on val/test samples). Non-overlaping if stride==interval_length
    :param features: list of features used
    :param valid_ids: list of ids used for validation
    :param test_ids: list of ids used for testing
    :param context_length: length of the additional context

    :return: values and labels for lvl1 and lvl2 for all participants
    """
    path = "../../data/"+ df_name
    data = {}

    #part_list = [1, 10, 11, 12, 14, 15, 16, 18, 19,  2, 20, 21, 22, 23, 24, 25, 26,
    #   28,  3,  4,  5,  6,  7,  8,  9]
    
    df = pd.read_csv(path)
    #transform participant ids to be in ascendent order from 0 to n_p
    participants_uniques = df['participant'].unique()
    dict_participant = {participants_uniques[i]: i for i in range(len(participants_uniques))}
    #print(dict_participant)
    df['participant'] = df['participant'].map(dict_participant)
    participants_uniques = df['participant'].unique()
    #print(participants_uniques)

    #select only data for participant
    df_p = df[df["participant"] == participant]
    #shuffle the rows
    df_p = df_p.sample(frac=1).reset_index(drop=True)
    feature_modalities = modalities_combination_data_prep(config.modalities_combination, df_p, config.feature_set_tag)# config.groundtruth)

    init_cols = df_p.loc[:, ['frame','participant','binary_label','multiclass_label']]
    df_p = pd.concat([init_cols, feature_modalities], axis=1, ignore_index=True)
    #column names
    df_p.columns = ['frame','participant','binary_label','multiclass_label'] + list(feature_modalities.columns)
    if config.dataset_processing == "pca":
        df = apply_pca(df_p)
   

    #data[participant]= df_p
    #print(data[participant].shape)

    lvl1_data = []

    minimum_training_samples = int(participant_minimum*df_p.shape[0])
    print('MINIMUM TRAINING SAMPLES')
    print(minimum_training_samples)
    #validation is 20% of the data 
    #test is remaining.
    val_samples = int(0.2*minimum_training_samples)
    data['train'] = df_p.iloc[:minimum_training_samples]
    data['val'] = df_p.iloc[minimum_training_samples:minimum_training_samples+val_samples]
    data['test'] = df_p.iloc[minimum_training_samples+val_samples:]

    for split in ['train', 'val', 'test']:
        values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[split], interval_length=interval_length, stride=stride_eval, features=features, context_length=context_length)
        lvl1_data.append((values, labels, raw_labels))
                 

    print('train',len(lvl1_data[0][0]))
    print('val',len(lvl1_data[1][0]))
    print('test',len(lvl1_data[2][0]))
    print(data['train'].shape)

    return lvl1_data, data


def evaluate_pp(config, group, name, valid_preds, y_val, test_preds, y_test):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {}
    results["val_accuracy"] = accuracy_score(y_val, valid_preds)
    cm_val = confusion_matrix(y_val, valid_preds)
    print(classification_report(y_val, valid_preds))
    results["0 count val"] = (y_val == 0).sum()


    #for confusion matrix, get the labels to understand which label it is
    results["val_precision"], results["val_recall"], results["val_fscore"], results["val_support"] = precision_recall_fscore_support(
        y_val, valid_preds, average=None)
    results["val_macroF1"] = f1_score(
        y_val, valid_preds, average="macro")
    print("VAL CM", cm_val)
    results['val_cm'] = cm_val

    results["test_accuracy"] = accuracy_score(y_test, test_preds)
    cm_test = confusion_matrix(y_test, test_preds)
    results["0 count test"] = (y_test == 0).sum()
    print(classification_report(y_test, test_preds))
    results["test_precision"], results["test_recall"], results["test_fscore"], results["test_support"] = precision_recall_fscore_support(
        y_test, test_preds, average=None)
    results["test_macroF1"] = f1_score(
        y_test, test_preds, average="macro")
    results["test_cm"] = cm_test    
        # wandb.log(results)

    results["y_val"]   = y_val
    results["y_test"]  = y_test
    results["valid_preds"] = valid_preds
    results["test_preds"] = test_preds

    #using get_metrics
    #validation
    print('VALIDATION SET')
    print(len(y_val), len(valid_preds))
    results_val = get_metrics(y_val, valid_preds, tolerance=1)

    #change results_val such that keys have "Val" in the start
    results_val = {f"Val_{k}": v for k, v in results_val.items()}

    print(results_val)
    #testing
    print('TEST SET')
    print(len(y_test), len(test_preds))
    results_test = get_metrics(y_test, test_preds, tolerance=1)
    results_test = {f"Test_{k}": v for k, v in results_test.items()}
    print(results_test)

    return results, results_val, results_test   
