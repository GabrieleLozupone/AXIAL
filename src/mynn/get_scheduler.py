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
from torch.optim import lr_scheduler as schedulers


def get_schedul(scheduler_name):
    if scheduler_name not in ["ReduceLROnPlateau", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR"]:
        raise NotImplementedError(f"Invalid scheduler name: {scheduler_name}")
    if scheduler_name == "ReduceLROnPlateau":
        return schedulers.ReduceLROnPlateau
    elif scheduler_name == "CosineAnnealingLR":
        return schedulers.CosineAnnealingLR
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return schedulers.CosineAnnealingWarmRestarts
    elif scheduler_name == "StepLR":
        return schedulers.StepLR
