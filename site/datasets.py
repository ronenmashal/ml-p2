import wandb
from collections import namedtuple
import numpy as np
import os

Dataset = namedtuple("Dataset", ["images", "labels"])
dataset_names = ["training", "validation", "test"]

def read_datasets(wandb_run, dataset_tag = "latest"):
    artifact = wandb_run.use_artifact(f'ml-p2/ml-p2/fashion-mnist:{dataset_tag}', type='dataset')
    data_dir = artifact.download()
    return [ read_dataset(data_dir, ds_name) for ds_name in dataset_names ]

def read_dataset(data_dir, ds_name):
    filename = ds_name + ".npz"
    data = np.load(os.path.join(data_dir, filename))
    return Dataset(images = data["x"], labels = data["y"])
