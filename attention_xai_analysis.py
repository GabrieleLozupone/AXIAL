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

import torch
from src.explainability.attention.utils import inference
from src.explainability.attention.utils import save_histograms_folds
from src.explainability.attention.utils import entire_dataset_histogram
from src.explainability.attention.utils import generate_explainable_mri
from src.explainability.attention.utils import compute_xai_metrics


def main():
    model_name = "Axial3DVGG16"
    # Run the inference function
    print(f"Running inference for {model_name}...")
    inference(model_name=model_name)
    # Run the plot function
    print("Saving histograms of 5 folds...")
    save_histograms_folds(model_name=model_name, percentile=74, tollerance=3)
    # Run the plot function for the entire dataset
    print("Saving histogram of entire dataset...")
    entire_dataset_histogram(model_name=model_name, percentile=74, tollerance=3)
    # Generate the explainable MRI
    print("Generating explainable MRI...")
    generate_explainable_mri(model_name=model_name, fold_num="entire_dataset", amplification_factor=10)
    # Compute the XAI metrics
    print("Computing XAI metrics...")
    compute_xai_metrics(model_name=model_name, fold_num="entire_dataset", percentile=99.9)


if __name__ == "__main__":
    # Print the PyTorch version used
    print(f"PyTorch version: {torch.__version__}")
    # Run the main function
    main()
