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
import torch.optim as optim
from lion_pytorch import Lion


def get_optim(optimizer_name):
    if optimizer_name not in ["Adam", "AdamW", "SGD", "Lion"]:
        raise NotImplementedError(f"Invalid optimizer name: {optimizer_name}")
    if optimizer_name == "Adam":
        return optim.Adam
    elif optimizer_name == "AdamW":
        return optim.AdamW
    elif optimizer_name == "SGD":
        return optim.SGD
    elif optimizer_name == "Lion":
        return Lion
