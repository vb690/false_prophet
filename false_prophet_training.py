from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler as mms

import matplotlib.pyplot as plt

from modules.oracles import FalseProphet
from modules.utils.general_utils import save_obj, load_obj

###############################################################################

for data_source in ['female_birth', 'peyton_manning', 'co2_daily']:

    splitted_arrays = load_obj(
        path=f'data//arrays//{data_source}'
    )
    scaler = mms()
    splitted_arrays = splitted_arrays['train']

    t_steps = splitted_arrays['y'].shape[1]
    scaler.fit(splitted_arrays['y'].reshape(-1, 1))
    t_steps = splitted_arrays['y'].shape[1]
    splitted_arrays['y'] = scaler.transform(
        splitted_arrays['y'].reshape(-1, 1)
    ).reshape(-1, t_steps, 1)
    splitted_arrays['target'] = scaler.transform(
        splitted_arrays['target'].reshape(-1, 1)
    ).reshape(-1, t_steps, 1)

    save_obj(
        obj=scaler,
        path=f'results//saved_obj//{data_source}_rsc'
    )

    val_point = int(splitted_arrays['y'].shape[0] * 0.2)

    X = [
        splitted_arrays['y'][:-val_point, :],
        splitted_arrays['month'][:-val_point, :],
        splitted_arrays['day_week'][:-val_point, :],
        splitted_arrays['day_month'][:-val_point, :]
    ]
    y = splitted_arrays['target'][:-val_point, :]

    X_val = [
        splitted_arrays['y'][-val_point:, :],
        splitted_arrays['month'][-val_point:, :],
        splitted_arrays['day_week'][-val_point:, :],
        splitted_arrays['day_month'][-val_point:, :]
    ]
    y_val = splitted_arrays['target'][-val_point:, :]

###############################################################################

    ES = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=100,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    model = FalseProphet(
        X_shape=splitted_arrays['y'][:-val_point, :].shape,
        batch_size=1,
        model_tag=data_source
    )
    model.build()

###############################################################################

    history = model.fit(
        X=X,
        y=y,
        verbose=1,
        validation_data=(X_val, y_val),
        epochs=10000,
        callbacks=[ES]
    )

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    model.save_weights()

###############################################################################
