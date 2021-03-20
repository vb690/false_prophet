import os

import pandas as pd

from modules.utils.data_utils import generate_data, split_data
from modules.utils.general_utils import save_obj

for csv_file in os.listdir('data//csv//'):

    df = pd.read_csv(
        f'data//csv//{csv_file}',
        sep=None
    )
    df = df.dropna(axis=1)

    arrays, remappers = generate_data(
        df=df,
        date_column='ds',
        value_column='y'
    )

    save_obj(
        obj=remappers,
        path=f'results//saved_obj//{csv_file[:-4]}_rmp'
    )

    splitted_arrays = split_data(
        arrays,
        test_size=0.2
    )

    save_obj(
        obj=splitted_arrays,
        path=f'data//arrays//{csv_file[:-4]}'
    )
