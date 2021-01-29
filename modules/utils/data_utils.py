import numpy as np

import pandas as pd

from .general_utils import find_factors


def find_time_steps(iterable):
    """
    """
    terminated = False
    while not terminated:

        factors = find_factors(len(iterable))
        batches = [len(iterable) / factor for factor in factors]
        print(f'Bacthes are {batches}')
        choice = int(input('Chose a factor: '))
        if choice == -1:
            iterable = iterable[:-1]
            print('Dropping one time step.')
        elif choice not in factors:
            print('Must chose a valid factor')
        else:
            terminated = True

    return choice


def generate_remappers(df, columns):
    """
    """
    remappers = {}
    for column in columns:

        remappers[column] = {
            value: code for code, value in enumerate(df[column].unique())
        }

    return remappers


def generate_arrays(df, columns, remappers):
    """
    """
    arrays = {}
    for column in columns:

        values = df[column].values
        values = values[:-1]
        time_steps = find_time_steps(values)
        if column in remappers:
            values = np.vectorize(remappers[column].get)(values)
            values = values.reshape(-1, time_steps)
        else:
            values = values.reshape(-1, time_steps, 1)
        arrays[column] = values

    return arrays


def generate_data(df, date_column, value_column):
    """
    """
    df = df.sort_values(date_column)

    df[value_column] = df[value_column].astype('float32')

    df['month'] = pd.DatetimeIndex(df[date_column]).month
    df['month'] = df['month'].shift(-1)

    df['day_week'] = pd.DatetimeIndex(df[date_column]).dayofweek
    df['day_week'] = df['day_week'].shift(-1)

    df['day_month'] = pd.DatetimeIndex(df[date_column]).day
    df['day_month'] = df['day_month'].shift(-1)

    df['target'] = df[value_column].shift(-1)

    remappers = generate_remappers(
        df=df,
        columns=[
            'month',
            'day_week',
            'day_month'
        ]
    )

    arrays = generate_arrays(
        df=df,
        columns=[
            'y',
            'month',
            'day_week',
            'day_month',
            'target'
        ],
        remappers=remappers
    )

    return arrays, remappers


def split_data(arrays, test_size=0.2):
    """
    """
    splitted_arrays = {
        'train': {},
        'test': {}
    }
    for name, array in arrays.items():

        t_steps = array.shape[0]
        cut_point = int(t_steps * test_size)

        splitted_arrays['train'][name] = array[:-cut_point, :]
        splitted_arrays['test'][name] = array[-cut_point:, :]

    return splitted_arrays
