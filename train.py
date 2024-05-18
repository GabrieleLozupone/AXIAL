# --------------------------------------------------------------------------------
# Copyright (c) 2024 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
# 
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
# 
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------
from datetime import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from src.models.aware_net import run_aware_experiment
from src.data import train_val_test_subject_split
from src.models import compute_metrics
from src.logger import create_writer
from src.data import load_dataframe
from src.config import load_config
from src.utils import get_device
from src.utils import set_seeds
from src import experiments


def main():
    # Get the timestamp for the experiment
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Load the yaml config file
    config = load_config('config.yaml')
    # Get the device to use
    device = get_device(cuda_idx=config['cuda_device'])
    # Get the dataframes and subjects based on the classification task
    df, subjects = load_dataframe(config['dataset_csv'], config['task'])
    # Print information about the subjects (number of subjects, number of subjects with each diagnosis)
    print(f"Number of subjects: {len(subjects)}")
    # Define k-fold cross validation
    kf = KFold(n_splits=config['k_folds'],
               shuffle=True,
               random_state=config['random_seed'])
    # Create lists to store the metrics for each fold
    fold_metrics = []
    all_y_true = torch.tensor([])
    all_y_pred = torch.tensor([])
    # Perform cross-validation on the dataset based on the subjects
    for train_val_subj_index, test_subj_index in kf.split(subjects):
        print(f"\n\n ----------------------- Fold {len(fold_metrics) + 1} ----------------------- \n\n")
        # Split the dataset in train, val and test by subject
        train_df, val_df, test_df = train_val_test_subject_split(df=df,
                                                                 train_val_subj=subjects[train_val_subj_index],
                                                                 test_subj=subjects[test_subj_index],
                                                                 val_perc_split=config['val_perc_split'],
                                                                 random_seed=config['random_seed'])
        # Set seeds for reproducibility
        set_seeds(config['random_seed'])
        # Instantiate the tensorboard writer for this fold
        writer = create_writer(
            config=config,
            fold_num=len(fold_metrics) + 1,
            timestamp=timestamp,
            extra=config["extra"]
        )
        # Run the experiment
        if config['network_3D'] == 'AwareNet' and config['image_type'] == '3D':
            y_true, y_pred, test_metrics = run_aware_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                config=config,
                writer=writer,
                device=device,
                fold=len(fold_metrics) + 1
            )
        else:
            y_true, y_pred, test_metrics = experiments.run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                config=config,
                writer=writer,
                device=device,
                fold=len(fold_metrics) + 1
            )
        # Concatenate the true and predicted labels for each fold
        all_y_true = torch.cat((all_y_true, y_true), dim=0)
        all_y_pred = torch.cat((all_y_pred, y_pred), dim=0)
        fold_metrics.append(test_metrics)
    # Calculate the mean of each metric
    mean_metrics = {
        metric: np.mean([entry[metric] for entry in fold_metrics])
        for metric in fold_metrics[0]  # Assuming all dictionaries have the same keys
    }
    # Save the fold results in json format with indent 4
    if config['skip_training']:
        with open(os.path.join(config['experiment_folder'], "results.json"), "w") as f:
            json.dump(fold_metrics, f, indent=4)
    else:
        with open(os.path.join(writer.get_logdir(), "../", "results.json"), "w") as f:
            json.dump(fold_metrics, f, indent=4)
    # Print the mean metrics
    print("\n\n------------- Mean metrics: -------------", end="\n\n")
    for metric, mean_value in mean_metrics.items():
        print(f"Mean {metric}: {mean_value}")
    # Compute the overall metrics
    overall_metrics = compute_metrics(y_true=all_y_true, y_pred=all_y_pred, num_classes=len(config['task']))
    # Save the overall metrics in json format with indent 4
    if config['skip_training']:
        with open(os.path.join(config['experiment_folder'], "overall_metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=4)
    else:
        with open(os.path.join(writer.get_logdir(), "../", "overall_metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=4)
    # Save the config in json format with indent 4
    if config['skip_training']:
        with open(os.path.join(config['experiment_folder'], "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(os.path.join(writer.get_logdir(), "../", "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    # Print the overall metrics
    print("\n\n------------- Overall metrics: -------------", end="\n\n")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value}")
    # Print the metrics for each fold
    print("\n\n------------- Metrics for each fold: -------------", end="\n\n")
    for i, fold_metric in enumerate(fold_metrics):
        print(f"Fold {i + 1}: {fold_metric}")


if __name__ == "__main__":
    # Print the PyTorch version used
    print(f"PyTorch version: {torch.__version__}")
    # Run the main function
    main()
