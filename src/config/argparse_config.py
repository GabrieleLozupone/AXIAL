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


def parse_arguments():
    parser = argparse.ArgumentParser("Train a 3D Attentional CNN on ADNI Complete 1Yr 1.5T dataset")
    parser.add_argument('--gpu_idx', type=int, help='specify gpu to utilize (default 0)', default=0)
    parser.add_argument('--backbone', type=str,
                        help='choose ResNet50 - EfficientNetV2S - EfficientNetV2M', required=True)
    parser.add_argument('--dataset', type=str, help='choose ImageNet - RadImageNet', required=True,
                        default="RadImageNet")
    parser.add_argument('--resume',
                        help='use pretrained weights. Expect to find the .pth in torch_models/--model_name.pth',
                        action="store_true")
    parser.add_argument('--batch_size', type=int, help='batch size (default 32)', default=32)
    parser.add_argument('--num_slices', type=int, help='number of slices to use (default 80)', default=80)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
    parser.add_argument('--freeze_first_perc', type=float, help='percentage of backbone to freeze starting from the '
                                                                'first layer', default=0.75)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--num_workers', type=int, help='num of threads used by DataLoader', default=10)
    parser.add_argument('--experiment_extra', type=str, help='extra string to append to the experiment name',
                        default="")
    args = parser.parse_args()
    return args
