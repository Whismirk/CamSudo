# -*- coding: utf-8 -*-
from operator import mod
import cnn

EPOCHS = 15
BATCH_SIZE = 128
SAVE_PATH = "models/model_mnist.h5"

model = cnn.build_model()
model, history = cnn.train_model(model, BATCH_SIZE, EPOCHS)
cnn.save_model(model, SAVE_PATH)
cnn.evaluate_model(history)
