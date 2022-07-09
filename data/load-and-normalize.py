import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from datasets import Dataset, log_datasets

# Setting random seeds to ensure predictable output.
random.seed(49)
np.random.seed(31)
tf.random.set_seed(19)

n_labels = 10


def load_mnist():
    # load the data from kera fashion_mnist.
    (train_images, train_labels),(test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    validation_images = train_images[:5000]
    validation_labels = train_labels[:5000]
    train_images = train_images[5000:]
    train_labels = train_labels[5000:]

    training_set = Dataset(train_images, train_labels)
    validation_set = Dataset(validation_images, validation_labels)
    test_set = Dataset(test_images, test_labels)

    return [training_set, validation_set, test_set]

def normalize_images(dataset):
    images, labels = dataset.images, dataset.labels
    images = images / 255.0
    return Dataset(images, labels)



# load everything from scrach and preprocess
datasets = [ normalize_images(ds) for ds in load_mnist() ]

# save the datasets
log_datasets("ml-p2", datasets, "normalize_images", "Fashion MNIST, Normalized.")