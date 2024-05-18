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
import argparse
from bids import BIDSLayout
from src.preprocessing import process_sMRIs
from src.utils import create_dataset_csv_from_bids, create_conversion_dataset_csv_from_bids


def main():
    parser = argparse.ArgumentParser("Preprocess the MRI images from the ADNI dataset")
    parser.add_argument('--bids_path', type=str, help='path to the BIDS dataset', required=True)
    parser.add_argument('--n_proc', type=int, help='number of processes to use', default=10)
    parser.add_argument('--checkpoint', type=str, help='path to the checkpoint file', default='checkpoint.txt')
    args = parser.parse_args()
    # Path to the dataset
    dataset_path = args.bids_path
    # Create a BIDSLayout object pointing to the dataset
    layout = BIDSLayout(dataset_path)
    # Get all the images
    image_files = layout.get(extension='nii.gz', suffix='T1w')
    print(f"Processing {len(image_files)}images...")
    # Process the images
    process_sMRIs(image_files, args.checkpoint, n_proc=args.n_proc)
    # Create the dataset csv file to be used for loading the data for classes (AD, CN, MCI)
    print("Creating dataset.csv file...")
    create_dataset_csv_from_bids(dataset_path)
    # Create the csv file for sMCI vs pMCI progression task
    print("Creating dataset_conversion.csv file...")
    create_conversion_dataset_csv_from_bids('data/ADNI1_Complete_1_Yr_1.5T/ADNI_BIDS/', 36)


if __name__ == '__main__':
    main()
