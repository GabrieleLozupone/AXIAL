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
import pytorch_warmup as warmup


def get_warmup_schedul(scheduler_name, optimizer, kwargs):
    if scheduler_name == 'LinearWarmup':
        print('Using LinearWarmup')
        return warmup.LinearWarmup(optimizer, **kwargs)
    elif scheduler_name == 'ExponentialWarmup':
        return warmup.ExponentialWarmup(optimizer, **kwargs)
    else:
        return None
