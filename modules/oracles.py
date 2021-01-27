from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense

from .utils.models_utils import _AbstractEstimator


class FalseProphet(_AbstractEstimator):
    """
    """
    def __init__(self, X, prob=True, model_tag=None, batch_size=256,
                 dropout=0.1):
        """
        """
        if model_tag is None:
            self.model_tag = 'FalseProphet'
        else:
            self.model_tag = model_tag
        self.prob = prob
        self.X = X.shape
        self.batch_size = batch_size

    def build(self, compile_schema=None):
        """
        """
        # COMPILE SCHEMA
        if compile_schema is None:
            compile_schema = {
                'month': 30,
                'day_week': 30,
                'day_month': 30,
                'env_layers': (100, 70,  50),
                'recurrent_layers': (50,),
                'regressor_layers': (50, 25),
                'dropout': 0.2,
                'activation': 'relu',
                'prob': True
            }

        for key, value in compile_schema.items():

            setattr(self, key, value)

        lag_input = Input(
            name='lag_input',
            batch_shape=(self.batch_size, self.X[1], self.X[2])
        )
        month_input = Input(
            name='month_input',
            batch_shape=(self.batch_size, self.X[1])
        )
        day_week_input = Input(
            name='day_week_input',
            batch_shape=(self.batch_size, self.X[1])
        )
        day_month_input = Input(
            name='day_month_input',
            batch_shape=(self.batch_size, self.X[1])
        )

        # EMBEDDINGS
        month_embedding = self._generate_embedding_block(
            input_tensor=month_input,
            input_dim=13,
            tag='month',
            prob=self.prob,
            activation=self.activation,
            output_dim=self.month
        )
        day_week_embedding = self._generate_embedding_block(
            input_tensor=day_week_input,
            input_dim=7,
            tag='day_week',
            prob=self.prob,
            activation=self.activation,
            output_dim=self.day_week
        )
        day_month_embedding = self._generate_embedding_block(
            input_tensor=day_month_input,
            input_dim=32,
            tag='day_month',
            prob=self.prob,
            activation=self.activation,
            output_dim=self.day_month
        )

        # FEATURES
        features = Concatenate(
            name='cocncatenation'
        )(
            [
                lag_input,
                month_embedding,
                day_week_embedding,
                day_month_embedding,
            ]
        )
        features = self._generate_fully_connected_block(
            input_tensor=features,
            tag='features_fully_connect',
            prob=self.prob,
            layers=self.env_layers,
            activation=self.activation
        )
        # RECURRENCY
        features = self._generate_recurrent_block(
            input_tensor=features,
            tag='features_recurrency',
            layers=self.recurrent_layers
        )
        # REGRESSOR
        regressor = self._generate_fully_connected_block(
            input_tensor=features,
            tag='regressor',
            prob=self.prob,
            layers=self.regressor_layers,
            activation=self.activation
        )

        # ESTIMATION
        estimation = Dense(
            1,
            name='estimation_dense'
        )(regressor)
        estimation = Activation(
            'relu',
            name='estimation'
        )(estimation)

        model = Model(
            inputs=[
                lag_input,
                month_input,
                day_week_input,
                day_month_input,
            ],
            outputs=estimation
        )
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae']
        )

        setattr(self, '_model', model)
