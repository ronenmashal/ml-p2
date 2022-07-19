import os
from datetime import datetime
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import wandb_helpers as wbh


def create_model(l1_size = 100, dropout_rate = 0.2, opt_lr = 0.01, opt_momentum=0.9, **more_args):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(l1_size, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dropout(rate = dropout_rate),
            keras.layers.Dense(l1_size, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dense(10, activation='softmax')	
        ])
        
    # compile the model
    opt = keras.optimizers.SGD(learning_rate=opt_lr, momentum=opt_momentum)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_hyperparameters_defaults(): 
    '''
    Use this method to initialize a W&B configuration structure.
    '''
    return {
        "activation" : "relu", 
        "l1_size" : 100,
        "dropout_rate" : 0.2,
        "opt_lr" : 0.01,
        "opt_momentum" : 0.9,
        "epochs" : 20
    }

model_name = "Conv"
model_description =  "Convolution model"
config = get_hyperparameters_defaults()

es_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode="auto",
    baseline=0.35,
    restore_best_weights=True,
)

with wbh.start_wandb_run(model_name, config) as run:
    train_set, validation_set, test_set = wbh.read_datasets(run)

    config = run.config
    model = create_model(**config)
    #model = keras.models.load_model(".\\saved_models\\Conv-084843")
    print (model.summary())

    model.fit(train_set.images, train_set.labels, 
        validation_data = (validation_set.images, validation_set.labels), 
        epochs = config["epochs"], 
        callbacks = [
            WandbCallback(),
            es_callback
        ])

    
    # model.save(f".\\saved-models\\{run.name}")
    # train_evaluation = model.evaluate(train_set.images, train_set.labels)
    # test_evaluation = model.evaluate(test_set.images, test_set.labels)

    #print("Train evaluation:", train_evaluation)
    #print("Test evaluation:", test_evaluation)

    #wbh.save_model(run, model, config, "Conv1", model_description)