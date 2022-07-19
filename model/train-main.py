'''
This is the place where we will train the models. It loads the dataset,
configure W&B and eventually save the trained weights as checkpoints. 
'''

from datetime import datetime
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import wandb_helpers as wbh

def create_model(input_shape = (28, 28, 1), class_count = 10, dropout_rate = 0.2, activation = "relu", l1_size = 200, l2_size = 100, l3_size = 60, l4_size = 60, **more_args):
    model = keras.Sequential(
        [
            keras.layers.Input(shape = input_shape),
            keras.layers.Flatten(input_shape = input_shape),
            keras.layers.Dense(l1_size, activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(l2_size, activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(l3_size, activation=activation),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(l4_size, activation=activation),
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
        "l1_size" : 200,
        "l2_size" : 100,
        "l3_size" : 60,
        "l4_size" : 60
    }

def get_best_hp_from_first_sweep():
    return {
        "input_shape": (28, 28, 1), 
        "class_count" : 10,
        "dropout_rate" : 0.055, 
        "activation" : "relu", 
        "l1_size" : 155,
        "l2_size" : 144,
        "l3_size" : 63
    }


n_labels = 10

config = get_hyperparameters_defaults()

model_name = "FCNN"
model_description =  "Simple fully-connected model, with 4 hidden layers."

config["epochs"] = 10

with wbh.start_wandb_run(model_name, config) as run:
    train_set, validation_set, test_set = wbh.read_datasets(run)

    print (len(validation_set.images))

    config = wandb.config

    model = create_model(**config)    
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics = ['accuracy'])
    print (model.summary())

    model.fit(train_set.images, train_set.labels, 
        validation_data = (validation_set.images, validation_set.labels), 
        epochs = config["epochs"], 
        callbacks = [
            WandbCallback()
        ]
    )

    train_evaluation = model.evaluate(train_set.images, train_set.labels)
    test_evaluation = model.evaluate(test_set.images, test_set.labels)

    print("Train evaluation:", train_evaluation)
    print("Test evaluation:", test_evaluation)

    #wbh.save_model(run, model, config, model_name, "Trained FCNN model with best configuration found by the sweep.")
    