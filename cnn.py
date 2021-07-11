# -*- coding: utf-8 -*-
from random import shuffle
import numpy
from tensorflow import keras
from matplotlib import pyplot

# Il y a, logiquement, 10 chiffres différents, de 0 à 9.
NB_CLASSES = 10
# Pour respecter les dimensions du set. 
# Chaque image du set a une largeur et hauteur de 28, ainsi qu'un seul canal de couleur.
SHAPE = (28, 28, 1)

# Récupération du dataset, ici MNIST, une base d'images de chiffres manuscrits directement intégrée à Keras.
def load_dataset():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    
    train_x = train_x.astype("float32") / 255
    test_x = test_x.astype("float32") / 255

    train_x = numpy.expand_dims(train_x, -1)
    test_x = numpy.expand_dims(test_x, -1)

    train_y = keras.utils.to_categorical(train_y, NB_CLASSES)
    test_y = keras.utils.to_categorical(test_y, NB_CLASSES)
    return train_x, train_y, test_x, test_y

# Compilation du modèle
def build_model():
        model = keras.models.Sequential()

        model.add(keras.Input(shape=SHAPE))

        model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(NB_CLASSES, activation="softmax"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

# Entrainement du modèle.
def train_model(model, batch_size, epochs):
    train_x, train_y, test_x, test_y = load_dataset()

    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), shuffle=1)

    return model, history

# On enregistre le modèle dans path.
def save_model(model, path):
    model.save(path, overwrite=True)

# Je vous pique cette fonction, désolé !
def evaluate_model(history):
    
    pyplot.style.use("ggplot")

    pyplot.plot(history.history["loss"], label="train_loss") 
    pyplot.plot(history.history["val_loss"], label="val_loss") 
    pyplot.plot(history.history["accuracy"], label="train_acc") 
    pyplot.plot(history.history["val_accuracy"], label="val_acc")

    pyplot.figure(1)
    pyplot.title("Training Loss and Accuracy")
    pyplot.xlabel("Epoch #")
    pyplot.ylabel("Loss/Accuracy")
    pyplot.legend()
    
    pyplot.show()