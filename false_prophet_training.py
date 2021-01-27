from tensorflow.keras.callbacks import EarlyStopping

from modules.utils.models_utils import save_full_model

from modules.oracles import FalseProphet
from modules.utils.general_utils import load_obj

###############################################################################

splitted_arrays = load_obj(
    path=f'data\\arrays\\peyton_manning'
)
splitted_arrays = splitted_arrays['train']

X = [
    splitted_arrays['transformed'][:-10, :],
    splitted_arrays['month'][:-10, :],
    splitted_arrays['day_week'][:-10, :],
    splitted_arrays['day_month'][:-10, :]
]

y = splitted_arrays['target'][:-10, :]

X_val = [
    splitted_arrays['transformed'][-10:, :],
    splitted_arrays['month'][-10:, :],
    splitted_arrays['day_week'][-10:, :],
    splitted_arrays['day_month'][-10:, :]
]
y_val = splitted_arrays['target'][-10:, :]

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
    X=splitted_arrays['transformed'][:-10, :],
    batch_size=1
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

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

model.save_weights()
save_full_model(model=model)

###############################################################################
