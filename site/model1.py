from datetime import datetime
import os
from time import time
from tensorflow import keras
import numpy as np

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

class VanillaNN(keras.Model):
    '''
    This is an implementation of a very simplified Full NN with 3 FC layers + Dropouts.
    It's used as a template/reference class for subsequent models, such that it provides
    certain methods used by the training script and the flask application.
    You might recognize the layers structure from Ex. 8...
    '''

    def __init__(self, input_shape, n_labels, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape = input_shape)
        self.hidden1 = keras.layers.Dense(200, activation='relu')
        self.hidden2 = keras.layers.Dense(100, activation='relu')
        self.hidden3 = keras.layers.Dense(60, activation='relu')
        self.output_layer = keras.layers.Dense(n_labels, activation='softmax')
        self._dropout_rate = dropout_rate
        self._name = "VanillaNN"
    
    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        input_layer = keras.layers.Dropout(rate=self.dropout_rate)(input_layer, training=training)
        hidden1 = self.hidden1(input_layer)
        hidden1 = keras.layers.Dropout(rate=self.dropout_rate)(hidden1, training=training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = keras.layers.Dropout(rate=self.dropout_rate)(hidden2, training=training)
        hidden3 = self.hidden3(hidden2)
        hidden3 = keras.layers.Dropout(rate=self.dropout_rate)(hidden3, training=training)
        output_layer = self.output_layer(hidden3)
        return output_layer
    
    def _get_dropout_rate(self):
        return self._dropout_rate
    
    def _get_name(self):
        return self._name

    dropout_rate = property(_get_dropout_rate)
    name = property(_get_name)

    @staticmethod
    def get_hyperparameters_defaults(): 
        '''
        Use this method to initialize a W&B configuration structure.
        '''
        return dict(
            learning_rate=0.001,
            epochs=20,
            dropout_rate = 0.2
        )

    @staticmethod
    def _create_model(dropout_rate):
        '''
        Instantiates and builds the model.
        '''
        model = VanillaNN((1, 28 * 28), 10, dropout_rate)
        model.build(input_shape=(1, 28 * 28))
        return model

    @staticmethod
    def create_model_for_training(config):
        '''
        Instantiates the model in 'training' mode, where dropout rate > 0. This means that in
        every ecpoch the Dropout layers drop out some of nodes.
        Also, for the purpose of training, the model is compiled with 'adam' optimization.
        '''
        model = VanillaNN._create_model(config["dropout_rate"])
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam')
        return model

    @staticmethod    
    def create_prediction_model(checkpoint_name):
        '''
        Instantiate and prepare the model for prediction: After the instantiation, the model loads a set
        of weights saved during training. The 'checkpoint_name' is the checkpoint suffix, provided during
        training.
        '''
        model = VanillaNN._create_model(0.0)
        checkpoint_path = model.get_checkpoint_path(checkpoint_name)
        model.load_weights(checkpoint_path)
        return model

    def get_checkpoint_path(self, suffix = ""):
        '''
        Get a checkpoint path ending with the given suffix.
        Use this to ensure consistency when defining checkpoints.
        '''
        current_path = os.path.dirname(os.path.abspath(__file__))
        cp_folder = os.path.join(current_path, "checkpoints", self.name, suffix)
        return cp_folder

    def predict_image_label(self, image):
        '''
        Predict a label based on a given image.
        The image must be an array of uint8, containing 28 * 28 = 784 items. 
        The shape does not matter. It will be reshaped to the necessary form.
        '''
        image = image / 255.0
        image = image.reshape([1, 28, 28])
        predictions = self.predict(image)
        label_idx = np.argmax(predictions[0])
        prediction = {
            "label id": label_idx,
            "label": class_names[label_idx],
            "certainty": predictions[0][label_idx]
        }
        return prediction

