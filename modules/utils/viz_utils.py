from sklearn.metrics import mean_absolute_error as mae

import matplotlib.pyplot as plt


def plot_performance(y, mean_prediction, upper_prediction, lower_prediction,
                     model='', dataset_name='', figsize=(10, 15)):
    """
    """
    error = mae(
        y,
        mean_prediction[:len(y)]
    )
    plt.figure(figsize=(10, 5))
    plt.plot(y, label='Ground Truth')
    plt.plot(
        mean_prediction,
        label='Out-of-sample Forecast',
        color='r'
    )
    plt.axvline(
        len(y),
        color='r',
        linestyle='--'
    )
    plt.fill_between(
        [i for i in range(len(mean_prediction))],
        upper_prediction,
        lower_prediction,
        color='r',
        alpha=0.1
    )
    plt.title(
        f'{model} Mean Absolute Error {round(error, 4)}'
    )
    plt.xlim(
        0,
        len(mean_prediction)
    )
    plt.ylabel('Y')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(
        f'results//plots//performance//{model}_{dataset_name}_performance.svg'
    )
    plt.show()
    return None
