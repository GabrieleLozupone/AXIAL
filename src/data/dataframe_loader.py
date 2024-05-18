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
import pandas as pd


def load_dataframe(csv_path, diagnosis):
    """
    Load the dataframe from a csv file and select only the subjects that have a diagnosis based on the task.
    @param csv_path: path to the csv file
    @param diagnosis: list of strings with the diagnosis to consider (e.g. ['CN', 'AD'])
    @return: dataframe and subjects that have a diagnosis based on the task
    """
    # Load the data
    df = pd.read_csv(csv_path)
    # Get only the MRI paths and the diagnosis with diagnosis CN and AD
    df = df[df['diagnosis'].isin(diagnosis)][['subject', 'mri_path', 'diagnosis']]
    # Get the df subjects
    subjects = df['subject'].unique()
    return df, subjects
