import pandas as pd

import numpy as np

from fbprophet import Prophet

from ..oracles import FalseProphet
from .general_utils import load_obj
from .viz_utils import plot_performance


def pred_stream(start_value, df_ts, model, remappers, rescaler,
                n_boot=30):
    """
    """
    df_ts = df_ts.sort_values('ds')
    month = pd.DatetimeIndex(df_ts['ds']).month.values
    day_week = pd.DatetimeIndex(df_ts['ds']).dayofweek.values
    day_month = pd.DatetimeIndex(df_ts['ds']).day.values

    start_value = rescaler.transform(
        np.array([start_value]).reshape(1, 1)
    ).reshape(1, 1, 1)

    month = np.array(
        [remappers['month'][value] for value in month]
    ).reshape(-1, 1)
    day_week = np.array(
        [remappers['day_week'][value] for value in day_week]
    ).reshape(-1, 1)
    day_month = np.array(
        [remappers['day_month'][value] for value in day_month]
    ).reshape(-1, 1)

    X = [
        start_value,
        month,
        day_week,
        day_month
    ]

    predictions = model.predict_with_uncertainty(
        X,
        n_boot=n_boot,
        batch_size=1
    )

    mean_prediction = predictions.mean(axis=1)
    mean_prediction = rescaler.inverse_transform(
        mean_prediction.reshape(-1, 1)
    ).flatten()
    upper_prediction = np.percentile(predictions, 92.5, axis=1)
    upper_prediction = rescaler.inverse_transform(
        upper_prediction.reshape(-1, 1)
    ).flatten()
    lower_prediction = np.percentile(predictions, 2.5, axis=1)
    lower_prediction = rescaler.inverse_transform(
        lower_prediction.reshape(-1, 1)
    ).flatten()

    return mean_prediction, upper_prediction, lower_prediction


def test_models(datasets, periods=None, figsize=(15, 5), n_boot=30):
    """
    """
    for dataset_name, cut_point in datasets.items():

        print(f'Testing {dataset_name} with Prophet')

        if periods is None:
            look_ahead = cut_point
        else:
            look_ahead = periods

        df = pd.read_csv(
            f'data//csv//{dataset_name}.csv'
        )

        df_tr = df[:-cut_point]

        y = df['y'].values[-cut_point:]
        remappers = load_obj(
            path=f'results//saved_obj//{dataset_name}_rmp'
        )
        rescaler = load_obj(
            path=f'results//saved_obj//{dataset_name}_rsc'
        )

        # ######################### PROPHET ###################################

        prophet = Prophet()
        prophet.fit(df_tr)

        df_ts = prophet.make_future_dataframe(periods=look_ahead)

        predictions = prophet.predict(df_ts)

        mean_prediction = predictions['yhat'].values[-look_ahead:]
        upper_prediction = predictions['yhat_upper'].values[-look_ahead:]
        lower_prediction = predictions['yhat_lower'].values[-look_ahead:]

        plot_performance(
            y=y,
            mean_prediction=mean_prediction,
            upper_prediction=upper_prediction,
            lower_prediction=lower_prediction,
            model='Prophet',
            dataset_name=dataset_name,
            figsize=figsize
        )

        # ################### FALSE PROPHET ###################################

        false_prophet = FalseProphet(
            X_shape=(None, 1, 1),
            batch_size=1,
            model_tag=dataset_name
        )

        false_prophet.build()
        false_prophet.load_weights()

        mean_prediction, upper_prediction, lower_prediction = pred_stream(
            start_value=df_tr['y'].values[-1],
            df_ts=df_ts[-look_ahead:],
            model=false_prophet,
            remappers=remappers,
            rescaler=rescaler,
            n_boot=n_boot
        )

        plot_performance(
            y=y,
            mean_prediction=mean_prediction,
            upper_prediction=upper_prediction,
            lower_prediction=lower_prediction,
            model='False Prophet',
            dataset_name=dataset_name,
            figsize=figsize
        )
