import pickle

from tqdm import tqdm

import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SpatialDropout1D
import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model

from .general_utils import generate_dir


class _AbstractEstimator:
    """
    """
    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_model']
        return state

    def _generate_embedding_block(self, input_tensor, input_dim,
                                  tag, prob=True, output_dim=20,
                                  activation='relu'):
        """
        """
        embedding = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            input_length=self.X[1],
            name=f'embedding_layer_{tag}'
        )(input_tensor)
        embedding = Activation(
            activation,
            name=f'embedding_activation_{tag}'
        )(embedding)

        if self.dropout is not None:
            embedding = SpatialDropout1D(
                self.dropout,
                name='sp_dropout_{}'.format(tag)
            )(embedding, training=prob)

        return embedding

    def _generate_fully_connected_block(self, input_tensor, tag, prob=True,
                                        layers=(50, 25), activation='relu'):
        """
        """
        for layer, units in enumerate(layers):

            if layer == 0:
                fully_connected = Dense(
                    units=units,
                    name=f'{layer}_dense_layer_{tag}'
                    )(input_tensor)
            else:
                fully_connected = Dense(
                    units=units,
                    name=f'{layer}_dense_layer_{tag}'
                    )(fully_connected)

            fully_connected = Activation(
                activation,
                name=f'{layer}_{activation}_activation_dense_layer_{tag}'
            )(fully_connected)

            if self.dropout is not None:
                fully_connected = SpatialDropout1D(
                    self.dropout,
                    name=f'sp_dropout_{layer}_{tag}'
                )(fully_connected, training=prob)

        return fully_connected

    def _generate_recurrent_block(self, input_tensor, tag, layers=(50,)):
        """
        """
        for layer, units in enumerate(layers):

            if layer == 0:
                recurrent = LSTM(
                    units=units,
                    return_sequences=True,
                    stateful=True,
                    batch_input_shape=K.int_shape(input_tensor),
                    name=f'{layer}_lstm_layer_{tag}'
                )(input_tensor)
            else:
                recurrent = LSTM(
                    units=units,
                    return_sequences=True,
                    stateful=True,
                    batch_input_shape=K.int_shape(input_tensor),
                    name=f'{layer}_lstm_layer_{tag}'
                )(recurrent)

        return recurrent

    def get_para_count(self):
        """
        """
        num_parameters = self.n_parameters
        return num_parameters

    def get_model(self):
        """
        """
        model = self._model
        return model

    def get_model_tag(self):
        """
        """
        model_tag = self.model_tag
        return model_tag

    def get_fitting_time(self):
        """
        """
        fitting_time = self.fitting_time
        return fitting_time

    def set_model(self, model):
        """
        """
        setattr(self, '_model', model)
        setattr(self, 'n_parameters', model.count_params())

    def fit(self, X, y, shuffle=False, **kwargs):
        """
        """
        start = time.time()
        history = self._model.fit(
            X,
            y,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **kwargs
        )
        end = time.time()
        setattr(self, 'fitting_time', end - start)
        return history

    def predict(self, X, **kwargs):
        """
        """
        t_steps = X[1].shape[0]
        predictions = []
        for t in range(t_steps):
            if t == 0:
                X_in = [
                    X[0],
                    X[1][t, :],
                    X[2][t, :],
                    X[3][t, :]

                ]
            else:
                X_in = [
                    np.array([predictions[t-1]]).reshape(1, 1, 1),
                    X[1][t, :],
                    X[2][t, :],
                    X[3][t, :]

                ]
            predictions.append(
                self._model.predict(X_in, batch_size=1).flatten()[0]
            )

        return np.array(predictions)

    def predict_with_uncertainty(self, X, n_boot=1, **kwargs):
        """
        """
        if not self.prob:
            print('Model is not using bayesian dropout')
            return None
        else:
            predictions = []
            for sample in tqdm(range(n_boot)):

                predictions.append(
                    self.predict(
                        X=X
                    )
                )

        return np.array(predictions).T

    def save_weights(self):
        """
        """
        name = self.model_tag
        self._model.save_weights(f'results//saved_training_weights//{name}.h5')
        return None

    def load_weights(self):
        """
        """
        name = self.model_tag
        self._model.load_weights(f'results//saved_training_weights//{name}.h5')
        return None


def save_full_model(model):
    """
    """
    name = model.get_model_tag()
    path = f'results//saved_training_models//{name}'
    generate_dir(path)

    keras_model = model.get_model()
    tf.saved_model.save(
        keras_model,
        f'{path}//engine//'
    )

    with open(f'{path}//scaffolding.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_full_model(name, custom_objects=None, **compile_schema):
    """
    """
    path = f'results//saved_training_models//{name}'

    keras_model = load_model(
        f'{path}//engine//',
        custom_objects=custom_objects,
        compile=False
    )
    keras_model.compile(
        **compile_schema
    )
    with open(f'{path}//scaffolding.pkl', 'rb') as input:
        model = pickle.load(input)

    try:
        model.set_model(keras_model)
    except Exception:
        import types

        def set_model(self, model):
            setattr(self, '_model', model)
            setattr(self, 'n_parameters', model.count_params())

        model.set_model = types.MethodType(set_model, model)
        model.set_model(keras_model)

    return model
