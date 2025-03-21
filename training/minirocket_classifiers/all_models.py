import wandb
import numpy as np
import pandas as pd
import random
#from sklearn.model_selection import KFold

#import tensorflow as tf
from get_metrics import get_metrics
#import tsai


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI & SKTIME ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
#!pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
#!pip install sktime -U  >> /dev/null

import tsai

from tsai.basics import *
#import sktime
import sklearn
#my_setup(sktime, sklearn)
from tsai.models.MINIROCKET import *
from create_data_splits import create_data_splits, create_data_splits_ids

import datetime
from sklearn.model_selection import ParameterGrid
from TimeSeries_Helpers import *
from itertools import product

# Generate all combinations of 4 modalities in Boolean
modalities_combinations = list(product([True, False], repeat=3)) # ['audio', 'face', 'talk']
modalities_combinations = [comb for comb in modalities_combinations if any(comb)] # remove all False combination


param_grid_models = {'model': ['LSTM_FCN',"GRU_FCN", "InceptionTime","InceptionTimePlus","MiniRocket", "gMLP"]}

param_grid_lstm = {'n_epoch': [100],
              'dropout_LSTM_FCN': [0,0.8,0.2,0.5],
              'fc_dropout_LSTM_FCN': [0, 0.2,0.5,0.8],
              'n_estimators': [50,20,100],
              'stride_train': [1,15, 30,80],
              'stride_eval': [1,15, 30,80],
              'lr': [2e-4,0.01,0.001],
              'focal_loss': [False, True],
              "interval_length": [1,15, 30,80],
              "context_length": [0],
              'oversampling': [False],
              "batch_size": [256],
              "batch_tfms": [None],
              "dataset_processing": ["pca", "norm", "clean"],
              "feature_set_tag": ["Full","Stat","RF"],
              "balanced": [True,False],
              "modalities_combination": modalities_combinations,
              "label": ['binary_label','multiclass_label'],
              #"part_fusion": ['early','intermediate','late'],
              #"groundtruth": ['sign', 'multi'],

              }



#merge both
param_grid = {**param_grid_models, **param_grid_lstm}

print(param_grid)


param_grid = list(ParameterGrid(param_grid))

#print length
print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")


df_name = 'all_participants_0_3.csv'
#df_full = pd.read_csv('../../data/' + df_name)
#features = df_full.columns[4:]
#print('FEATURES', features)

#remove configs were stride is bigger than the interval length
new_param_grid = []
for i,grid_config in enumerate(param_grid):
    if not (grid_config["stride_train"] > grid_config["interval_length"] or grid_config["stride_eval"] > grid_config["interval_length"]):
        #print("Removed config: ", grid_config)
        #param_grid.remove(grid_config)
        #print('removed')
    #else:
        new_param_grid.append(grid_config)

param_grid = new_param_grid


print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")


#seed randomizer
#random.seed(42)
#np.random.seed(42)
#randomize param_grid
random.shuffle(param_grid)


for i, grid_config in enumerate(param_grid):
    if i >= 0:
        if True:
            print("Round:", i+1, "of", len(param_grid))
            print(grid_config)
            config = AttrDict(
                df_name = df_name,
                merged_labels=False,
                threshold=80,
                interval_length=grid_config["interval_length"],
                stride_train=grid_config["stride_train"],
                stride_eval=grid_config["stride_eval"],
                context_length=grid_config['context_length'],
                train_ids=[],
                valid_ids=[],
                test_ids=[],
                use_lvl1=True,
                use_lvl2=False,
                model=grid_config["model"],
                lr=grid_config["lr"],
                n_epoch=grid_config["n_epoch"],
                dropout_LSTM_FCN=grid_config["dropout_LSTM_FCN"],
                fc_dropout_LSTM_FCN=grid_config["fc_dropout_LSTM_FCN"],
                batch_tfms=grid_config["batch_tfms"],
                batch_size=grid_config["batch_size"],
                focal_loss=grid_config["focal_loss"],
                n_estimators=grid_config["n_estimators"],
                #features = features,
                oversampling=grid_config["oversampling"],
                undersampling=False,
                verbose=True,
                dataset = "openface",
                dataset_processing = grid_config["dataset_processing"],
                feature_set_tag=grid_config["feature_set_tag"],
                modalities_combination = grid_config["modalities_combination"],
                balanced = grid_config["balanced"],
                label = grid_config["label"]
                #part_fusion = grid_config["part_fusion"],
                #groundtruth = grid_config["groundtruth"],
                
            )

            cross_validate(val_fold_size=5, config=config,
                        group="all", name=str(grid_config))