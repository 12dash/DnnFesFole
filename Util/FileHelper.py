import os

import pandas as pd


def read_csv(path, cols=None, sampling=1):
    dataframe = pd.read_csv(path)
    if cols is not None:
        dataframe.columns = cols
    if sampling is not None:
        dataframe = dataframe.sample(frac=sampling)
    return dataframe


def read_file(path, cols=None, skiprows=None, delimiter=",", sampling=1):
    dataframe = pd.read_csv(path, delimiter=delimiter, skiprows=skiprows)
    if cols is not None:
        dataframe.columns = cols
    if sampling is not None:
        dataframe = dataframe.sample(frac=sampling)
    return dataframe

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
