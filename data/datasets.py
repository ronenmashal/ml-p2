import wandb
from collections import namedtuple
import numpy as np

Dataset = namedtuple("Dataset", ["images", "labels"])

def log_datasets(project_name, datasets, job_type, stage_description):
    names = ["training", "validation", "test"]

    with wandb.init(project = project_name, job_type = job_type) as run:
        raw_data = wandb.Artifact('fashion-mnist', type='dataset', 
            metadata={
                "source" : "keras.datasets.fashion_mnist",
                "description" : stage_description,
                "sizes" : [ len(dataset.images) for dataset in datasets ]
            })

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".npz", mode = "wb") as file:
                np.savez(file, x = data.images, y = data.labels)
        
        run.log_artifact(raw_data)