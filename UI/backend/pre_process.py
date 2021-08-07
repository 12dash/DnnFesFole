import pandas as pd
from Util.FileHelper import *


def read_data(dataset, dataframe):
    df = pd.DataFrame()
    data_dir = '../../Code/Data/'
    p_type = 'Classification' 

    if dataset == 'Iris':
        path = data_dir+'Iris/iris.data'
        df = read_file(path, cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    elif dataset == 'Traffic':
        path = data_dir+f'Traffic/{dataframe}'
        df = read_file(path, cols=['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'CLass'],
                        skiprows=1, sampling = None, delimiter='\t')
        p_type='Regression'

    return df, p_type