import time, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from ..data_utils import save_confusion
IMG_SIZE, BATCH, EPOCHS = 224, 32, 10

def train(train_dir, classes):
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255., rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, validation_split=0.2)

    train = gen.flow_from_directory(train_dir, target_size=(IMG_SIZE,IMG_SIZE),
                                    batch_size=BATCH, subset="training", shuffle=True)
    val   = gen.flow_from_directory(train_dir, target_size=(IMG_SIZE,IMG_SIZE),
                                    batch_size=BATCH, subset="validation", shuffle=True)

    model = models.Sequential([
        layers.Input((IMG_SIZE,IMG_SIZE,3)),
        layers.Conv2D(32,3,activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(64,3,activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(128,3,activation='relu'), layers.GlobalAvgPool2D(),
        layers.Dense(128,activation='relu'),
        layers.Dense(len(classes),activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    t0=time.time(); hist=model.fit(train,validation_data=val,epochs=EPOCHS,verbose=1)
    return model, hist.history, time.time()-t0

def predict(model, paths_batch):
    imgs = np.stack([
        tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(p, target_size=(IMG_SIZE,IMG_SIZE))
        ) for p in paths_batch]) / 255.
    return model.predict(imgs, verbose=0).argmax(axis=1)
