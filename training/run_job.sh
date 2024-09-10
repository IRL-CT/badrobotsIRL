#!/bin/bash
#SBATCH -J lstm_model                         # Job name
#SBATCH -o ./training_outputs/lstm_model_%j.out                  # output file (%j expands to jobID)
#SBATCH -e ./training_outputs/lstm_model_%j.err                  # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=sjl356@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32gb                           # server memory requested (per node)
#SBATCH -t 20:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=gpu       # Request partition
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed

source /share/apps/anaconda3/2020.11/bin/conda.sh
conda activate /share/ju/conda_virtualenvs/badrobotsirl/

python3 lstm_model_wandb.py
