import argparse
import os
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from models import CRNN
from tools.data_loader import ImageLoader


def start_training(log_dir, optimizer=Adam(lr=0.001, decay=1e-6), loss="categorical_crossentropy",
                   metrics=None):
    """
    Train the model
    :param log_dir: directory to log the reports
    :param optimizer: optimizer for model (default: Adam)
    :param loss: loss function for model (default: categorical_crossentropy)
    :param metrics: metrics for model (default: accuracy)
    :return: model checkpoint for the best epoch with regards to validation accuracy
    """
    if metrics is None:
        metrics = ["accuracy"]

    train_data = ImageLoader(os.path.join(args.data_path, "training.csv"))
    val_data = ImageLoader(os.path.join(args.data_path, "validation.csv"))

    # Training Callbacks
    checkpoint_filename = os.path.join(log_dir, "weights.{epoch:02d}.model")
    model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1, monitor="val_acc")
    csv_logger_callback = CSVLogger(os.path.join(log_dir, "log.csv"))
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")

    crnn = CRNN()
    model = crnn.build_model(train_data.get_input_shape())
    model.compile(optimizer, loss, metrics)

    history = model.fit_generator(
        train_data.get_data(),
        steps_per_epoch=train_data.get_num_files() // args.batch_size,
        epochs=args.num_epochs,
        callbacks=[model_checkpoint_callback, csv_logger_callback, early_stopping_callback],
        verbose=1,
        validation_data=val_data.get_data(should_shuffle=False),
        validation_steps=val_data.get_num_files() // args.batch_size,
        max_queue_size=args.batch_size,
        workers=4,
        use_multiprocessing=True
    )

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    plt.savefig(os.path.join(log_dir, "history.png"))

    # Evaluation on model with best validation accuracy
    best_epoch = np.argmax(history.history["val_acc"])
    print("Log files: ", log_dir)
    print("Best epoch: ", best_epoch+1)

    model_file_name = checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch))
    return model_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='data_path', default=os.getcwd(), required=True)
    parser.add_argument('--epochs', dest='num_epochs', default=25)
    parser.add_argument('--batch_size', dest='batch_size', default=128)
    args = parser.parse_args()

    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Logging to {}".format(log_dir))

    model_filename = start_training(log_dir)
