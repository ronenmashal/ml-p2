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

config = TrainedModel.get_hyperparameters_defaults()
model = TrainedModel.create_model_for_training(config)


n_labels = 10

timestamp = datetime.now().strftime("%H%M%S")
cp_folder = model.get_checkpoint_path(timestamp)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_folder, save_weights_only=True, verbose=1)

with wandb.init(project=f"ml-p2", entity="ml-p2", name=f"train-{model.name}-{timestamp}" , 
    notes = f"Training model {model.name} @{timestamp}", config = config) as run:

    train_set, validation_set, test_set = read_datasets(run)

    config = wandb.config
    
    model.fit(train_set.images, train_set.labels, 
        validation_data = (validation_set.images, validation_set.labels), 
        epochs = config["epochs"], 
        callbacks = [WandbCallback(), cp_callback])

    train_evaluation = model.evaluate(train_set.images, train_set.labels)
    test_evaluation = model.evaluate(test_set.images, test_set.labels)

    print("Train evaluation:", train_evaluation)
    print("Test evaluation:", test_evaluation)


