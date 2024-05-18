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
import yaml


def load_config(path):
    """
    This function loads a yaml config file and returns a dictionary with the config parameters.
    @param path: path to the yaml config file
    @return: dictionary with the config parameters
    """
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"Loaded config file from {path}", end="\n\n")
            for key, value in config.items():
                print(f"{key}: {value}")
            return config
        except yaml.YAMLError as exc:
            print(exc)
