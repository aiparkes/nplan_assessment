"""
Data Loading Utilities
----------------------

:Author: Amy Parkes
:Date: 25/06/2021

Utilities for loading data from disk into memory.
"""

import json
import numpy as np
import os
from typing import Dict, Union


def load_np_files(set: str, data: Dict) -> Dict[str, np.array]:
    """
    Loads np files into an array, adds to dict 'data'

    :param set: The name of the data set
    :param data: Empty dictionary to populate
    :return: Dictionary with loaded np arrays as values
    """
    for np_file in ["_feats", "_labels", "_graph_id"]:
        np_path = os.path.join("data", set + np_file + ".npy")
        data.update({set + np_file: np.load(np_path)})
    return data


def load_json_files(set: str, data: Dict[str, np.array]) -> Dict[str, Union[np.array, Dict]]:
    """
    Loads json files into a dict, adds to dict 'data'

    :param set: The name of the data set
    :param data: Dictionary of np arrays to populate further
    :return: Diction with loaded np arrays and loaded jsons as values
    """
    json_path = os.path.join("data", set + "_graph.json")
    with open(json_path) as f:
        data.update({set + "_graphs": json.load(f)})
    return data


def load_data() -> Dict[str, Union[np.array, Dict]]:
    """
    Loads provided data into a dictionary
    """
    data = {}
    for set in ["train", "valid", "test"]:
        data = load_np_files(set, data)
        data = load_json_files(set, data)
    return data
