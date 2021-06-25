"""
Text Embedding Utilities
----------------------

:Author: Amy Parkes
:Date: 25/06/2021

Converts text to numeric values for ML training.
"""

import numpy as np
import pandas as pd
from typing import List

def text_to_feats(df: pd.DataFrame) -> List[np.array]:
    """
    Take pandas df with strings in column 'text', converts to character summary

    :param df: Pandas dataframe containing column 'text' with str values
    :return: list of 3 numeric columns
    """
    sum = []
    mean = []
    var = []
    for num, row in df.iterrows():
        ords = [ord(char) for char in row["text"]]
        sum.append(np.sum(ords))
        var.append(np.var(ords))
        mean.append(np.mean(ords))
    return [
        np.reshape(sum, (len(df), 1)),
        np.reshape(mean, (len(df), 1)),
        np.reshape(var, (len(df), 1)),
    ]
