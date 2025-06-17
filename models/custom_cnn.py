# models/custom_cnn.py

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from data_utils import save_confusion

# Parametry
IMG_SIZE = 224
BATCH    = 32
EPOCHS   = 100    # ustaw dowolnie dużą liczbę epok

# Upewnij się, że katalog na wyniki istnieje
os.makedirs("results", exist_ok=True)

def train(train_dir, classes):
    """
    Trenuje własną sieć CNN od zera na obrazach z `train_dir`,
    dzieli je na train/val (80/20) i zapisuje najlepsze wagi
    do plików results/cnn_epoch_<nr>_valloss_<val_loss>.weights.h5
    Zwraca: (model, historia, czas_treningu).
    """

    # Generacja danych z augmentacją i validacją
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_ds = gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        subset="training",
        shuffle=True
    )
    val_ds = gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        subset="validation",
        shuffle=True
    )

    # Definicja modelu
    model = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.GlobalAvgPool2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(classes), activation='softmax')
    ])


    # Callback do zapisu najlepszych wag (.weights.h5 wymagane przy save_weights_only=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath="results/cnn_epoch_{epoch:03d}_valloss_{val_loss:.4f}.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode="min"
    )

    # Trening
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    train_time = time.time() - t0

    return model, history.history, train_time



def predict(model, paths_batch):
    """
    Dla listy ścieżek wczytuje obrazy, normalizuje je i
    zwraca indeksy klas wybrane przez model.
    """
    imgs = np.stack([
        tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(
                p, target_size=(IMG_SIZE, IMG_SIZE)
            )
        ) for p in paths_batch
    ]) / 255.0

    preds = model.predict(imgs, verbose=0)
    return preds.argmax(axis=1)
