'''
This is the place where we will train the models. It loads the dataset,
configure W&B and eventually save the trained weights as checkpoints. 
'''

from datetime import datetime
import os
from gc import callbacks
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import wandb
from wandb.keras import WandbCallback
import numpy as np
from datasets import read_datasets

# If you want to change the trained model replace this import with the module of
# the new model. Also, make sure it has the same static and private functions as VanillaNN.
#
from model1 import VanillaNN as TrainedModel

def create_model(input_shape = (28, 28, 1), class_count = 10, dropout_rate = 0.2, activation = "relu", hidden_layers_sizes = [200, 100, 60]):
    model = keras.Sequential(
        [
            keras.layers.Input(shape = input_shape),
            keras.layers.Flatten(input_shape = input_shape),
            keras.layers.Dense(hidden_layers_sizes[0], activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(hidden_layers_sizes[1], activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(hidden_layers_sizes[2], activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(class_count, activation='softmax')            
        ]) 
    return model

def get_hyperparameters_defaults(): 
        '''
        Use this method to initialize a W&B configuration structure.
        '''
        return {
            "input_shape": (28, 28, 1), 
            "class_count" : 10,
            "dropout_rate" : 0.2, 
            "activation" : "relu", 
            "hidden_layers_sizes" : [200, 100, 60]
        }

n_labels = 10

timestamp = datetime.now().strftime("%H%M%S")
config = get_hyperparameters_defaults()
model = create_model(**config)    
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics = ['accuracy'])

print (model.summary())

model_name = "FCNN"
model_description =  "Simple fully-connected model, with 3 hidden layers."


# Create a callback that saves the model's weights
#cp_folder = model.get_checkpoint_path(timestamp)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_folder, save_weights_only=True, verbose=1)

config["epochs"] = 10


with wandb.init(project=f"ml-p2", entity="ml-p2", name=f"{model_name}-{timestamp}" , 
    notes = f"Training FCNN model @{timestamp}", config = config) as run:

    train_set, validation_set, test_set = read_datasets(run)

    print (len(validation_set.images))

    config = wandb.config
    
    model_artifact = wandb.Artifact(model_name, type = "model", description=model_description, metadata= dict(config))
    model.fit(train_set.images, train_set.labels, 
        validation_data = (validation_set.images, validation_set.labels), 
        epochs = config["epochs"], 
        callbacks = [
            WandbCallback(),
            #, cp_callback
        ]
    )

    train_evaluation = model.evaluate(train_set.images, train_set.labels)
    test_evaluation = model.evaluate(test_set.images, test_set.labels)

    print("Train evaluation:", train_evaluation)
    print("Test evaluation:", test_evaluation)

    model_file = f'./saved-models/{model_name}.tf'
    tf.keras.models.save_model(model, model_file)
    model_artifact.add_dir(model_file)
    wandb.log_artifact(model_artifact)

