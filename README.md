<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/false_prophet/blob/main/figures/false_prophet_logo.svg">
<p align="center">
  <i>"Beware of false prophets, who come to you in  <br /> 
   sheep’s clothing, but inwardly are ravening wolves." <br /> 
   <b> Matthew 7:15 </b></i> 
</p>

# Motivation


# Features

# How to use
## False Prophet
<p align="center">
  <img width="900" height="400" src="https://github.com/vb690/false_prophet/blob/main/figures/false_prophet_architecture.svg">
<p align="center">

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
    model_tag=data_source
)
model.build(compile_schema=compile_schema)
```

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
False prophet uses [Monte Carlo drop-out for producing an approximated posterior distribution](https://arxiv.org/abs/1506.02142). For obtaining uncertainty in the model estimates we have to call the `predict_with_uncertainty` method specifying the number of samples that we want to draw from the posterior through the `n_boot` argument.
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
## False Prophet Training
## Models Comparison


# Performance Comparison

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
