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
from sklearn.model_selection import train_test_split


def train_val_test_subject_split(df,
                                 train_val_subj,
                                 test_subj,
                                 val_perc_split,
                                 random_seed):
    """
    Split the dataset in train, val and test by subject
    @param df: dataframe containing the dataset with columns ['subject', 'mri_path', 'diagnosis']
    @param train_val_subj: list of subjects to use for train and val
    @param test_subj: list of subjects to use for test
    @param val_perc_split: percentage of subjects in train_val_subj to use for validation
    @param random_seed: random seed to use for reproducibility
    @return: train_df, val_df, test_df
    """
    # Split the train_val subjects
    train_subjects, val_subjects = train_test_split(train_val_subj,
                                                    test_size=val_perc_split,
                                                    random_state=random_seed)
    # Split the dataset in train and test
    train_df = df[df['subject'].isin(train_subjects)]
    val_df = df[df['subject'].isin(val_subjects)]
    test_df = df[df['subject'].isin(test_subj)]
    return train_df, val_df, test_df
