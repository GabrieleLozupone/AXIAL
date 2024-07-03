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
from torch.utils.tensorboard import SummaryWriter
import os


def create_writer(config: dict,
                  fold_num: int,
                  timestamp: str,
                  extra: str = None) -> SummaryWriter:
    """
    Create a TensorBoard writer with a custom path.
    """
    hyperparams = 'rand_seed_' + str(config['random_seed']) + \
                  '_val_split_' + str(config["val_perc_split"]).lower() + \
                  '_epochs_' + str(config["num_epochs"]).lower() + \
                  "_lr_" + str(config["learning_rate"]).lower() + \
                  '_batch_size_' + str(config["batch_size"]) + \
                  '_dropout_' + str(config["dropout"]).lower() + \
                  '_wd_' + str(config["weight_decay"]).lower() + \
                  '_freeze_' + str(config["freeze_first_percentage"]).lower() + \
                  '_slices_' + str(config["num_slices"]).lower() + \
                  '_optim_' + str(config["optimizer"]).lower() + \
                  '_scheduler_' + str(config["scheduler"]).lower() + \
                  '_pretrained_' + str(config["load_pretrained_model"]).lower()
    if config["image_type"] == "3D":
        if config["network_3D"] == "Axial3D":
            model_name = os.path.join("Axial3D", config["backbone"])
        if config["network_3D"] == "MajorityVoting3D":
            model_name = os.path.join("MajorityVoting3D", config["backbone"])
        elif config["network_3D"] == "TransformerConv3D":
            model_name = os.path.join("TransformerConv3D", config["backbone"])
        elif config["network_3D"] == "AwareNet":
            model_name = "AwareNet"
    elif config["image_type"] == "2D":
        model_name = config["backbone"]
    else:
        raise NotImplementedError
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs",
                               config["dataset_name"],
                               ' vs '.join(sorted(config["task"])),
                               config["image_type"],
                               config["slicing_plane"],
                               model_name,
                               config["pretrained_on"],
                               hyperparams,
                               extra,
                               timestamp,
                               f"fold_{fold_num}")
    else:
        log_dir = os.path.join("runs",
                               config["dataset_name"],
                               ' vs '.join(sorted(config["task"])),
                               config["image_type"],
                               config["slicing_plane"],
                               model_name,
                               config["pretrained_on"],
                               hyperparams,
                               timestamp,
                               f"fold_{fold_num}")
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
