'''
This is the place where we will train the models. It loads the dataset,
configure W&B and eventually save the trained weights as checkpoints. 
'''

import datetime
import os
from gc import callbacks
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import wandb
from wandb.keras import WandbCallback

# If you want to change the trained model replace this import with the module of
# the new model. Also, make sure it has the same static and private functions as VanillaNN.
#
from model1 import VanillaNN as TrainedModel

config = TrainedModel.get_hyperparameters_defaults()
model = TrainedModel.create_model_for_training(config)


(train_images, train_labels),(test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
validation_images = train_images[:5000]
validation_labels = train_labels[:5000]

n_labels = len(set(validation_labels))

cp_folder = os.path.join(model.get_checkpoint_path(),  datetime.now().strftime("%H%M%S"))
#os.makedirs(cp_folder, exist_ok = True)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_folder, save_weights_only=True, verbose=1)

#wandb.init(project=f"P2-{model.name}", name = "Train", notes = "Training the model as taken from Ex 8.", config = config)
#config = wandb.config
#model.fit(train_images, train_labels, epochs = config["epochs"], callbacks = [WandbCallback(), cp_callback]) # input_type = "image", labels = train_labels

model.fit(train_images, train_labels, epochs = config["epochs"], callbacks = [cp_callback]) # input_type = "image", labels = train_labels

