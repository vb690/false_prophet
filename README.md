<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/figures/false_prophet_logo.svg">
<p align="center">
  <i>"Beware of false prophets, who come to you in  <br /> 
   sheep’s clothing, but inwardly are ravening wolves." <br /> 
   <b> Matthew 7:15 </b></i> 
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vb690/false_prophet/HEAD)

# Motivation

This repository aims to ompare the perfromance of [Facebook's Prophet](https://facebook.github.io/prophet/) with that of an Artificial Neural Network using recurrent operations (RNN) in an "out of the box" setting. We already expect the RNN to perfrorm worse both in terms of estimates error and training time given that Prophet has been engineered to be accurate and fast even wehn minimal inputs from the end user are provided.
  
However, there is value in seeing where and how the RNN fails given that ANNs have the potential to achieve bettter perfromance and generalizability if carefully tuned.

# Features

* False Prophet: a recurrent neural network based forecaster with seasonality and probabilitic output.
* Data preparation pipeline.
* False Prophet training pipeline.
* Models comparison pipeline.

# How to use
## False Prophet
<p align="center">
  <img width="900" height="400" src="https://github.com/vb690/false_prophet/blob/main/figures/false_prophet_architecture.svg">
<p align="center">
  
False Prohet is nothing more than an Artificial Neural Network employing statefull recurrent layers (LSTMs here) for modelling temporality. For instatiating the model we need to pass a `compile_schema` specifying hyperparamenters for various portions of the model.
```python
from tensorflow.keras.callbacks import EarlyStopping
from modules.oracles import FalseProphet

compile_schema = {
    'month': 20,
    'day_week': 20,
    'day_month': 20,
    'env_layers': (50,  25),
    'recurrent_layers': (100,),
    'regressor_layers': (50, 25),
    'dropout': 0.2,
    'activation': 'relu',
    'prob': True
}

model = FalseProphet(
    X_shape=(None, 10, 1),
    batch_size=1,
    model_tag='test_model'
)
model.build(compile_schema=compile_schema)
```
False Prophet wraps the conventional Keras `fit` and `predict` methods giving us the freedom to use callbacks like `EarlyStopping` during training.
```python
ES = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=100,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

history = model.fit(
    X=X_tr,
    y=y_tr,
    verbose=1,
    validation_data=(X_val, y_val),
    epochs=10000,
    callbacks=[ES]
)
```
False Prophet uses [Monte Carlo drop-out for producing an approximated posterior distribution](https://arxiv.org/abs/1506.02142). For obtaining uncertainty in the model estimates we have to call the `predict_with_uncertainty` method specifying the number of samples that we want to draw from the posterior through the `n_boot` argument.
```python
predictions = model.predict_with_uncertainty(
    X_ts,
    n_boot=50,
    batch_size=1
)

mean_prediction = predictions.mean(axis=1)
upper_prediction = np.percentile(predictions, 92.5, axis=1)
lower_prediction = np.percentile(predictions, 2.5, axis=1)
```
  
## Data Preparation
Place `.csv` files on which you to perfrom the comparison in `'data\\csv\\'`. The files need to have the following format  

| ds         | y  |
|------------|----|
| yyyy-dd-mm | 10 |
| yyyy-dd-mm | 11 |
| yyyy-dd-mm | 3  |

The data preparation pipeline will then iterate over all the files in `'data\\csv\\'` and preprocess them in a format suitable for trainining False Prophet. Date will be decomposed in month, day of the week, day of the month and ordinal coded. The target will be rescaled and turned into a format suitable for RNN training.  
  
The script will save input features and target locally as `.npy` files in `'data\\arrays\\'` along with rescaler and ordinal encoder objects in `'results\\saved_obj\\'`.

## False Prophet Training
The training pipeline will iterate over all the data sources in `'data\\arrays\\'`, train FalseProphet untill convergence is reached or untill no significant improvement is observed for 100 epochs. At this point the script will save the model weights locally as a `.h5` file in `'results\\saved_training_weights\\'`.
## Models Comparison
Once FalseProphept has been fitted on all the available datasets, simply call the `test_models` function using the dataset you want to conduct the comparison on.
```python
test_models(
    datasets={'co2_daily': 360}, 
    periods=720, 
    figsize=(15, 5),
    n_boot=30
)
```
The function `test_models` will fit Prophet on-the-go on the specified dataset while it will build a FalseProphet ex-novo and load the weights relative to the same dataset. The core arguments to this function are:

* datasets: a dictionary with datasets names as keys and cutting points as values. The cutting points specify which portion of the dataset is used for producing out-of-sample estimates.
* periods: an integer specifying the number of time-steps beyond the last entry of the dataset for which we want to produce an estimate (i.e. estimates for which we do not have a ground truth).
* n_boot: an integer specifying the number of samples that we want to draw from the posterior. This can be a quite lengthy process as it effectively run the Keras `predict()` method `n_boot` times.
# Performance Comparison
Values before the vertical dotted line indicate out-of-sample estimates for which we have a ground truth while we do not possess a ground truth for values after the dotted line.
Each model oprates on a rolling-prediction basis: given a sequence of timestaps and the ground truth at `t-1` the models will produce an estimate for time `t` which will then be used as an input for producing a prediction at time `t+1`. 
## Peyton-Manning

### Prophet

<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/Prophet_peyton_manning_performance.svg">
<p align="center">
  
### False Prophet
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/False%20Prophet_peyton_manning_performance.svg">
<p align="center">
  
## Female Birth

### Prophet
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/Prophet_female_birth_performance.svg">
<p align="center">
  
### False Prophet
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/False%20Prophet_female_birth_performance.svg">
<p align="center">
  
## Daily CO2

### Prophet
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/Prophet_co2_daily_performance.svg">
<p align="center">
  
### False Prophet
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/results/plots/performance/False%20Prophet_co2_daily_performance.svg">
<p align="center">
