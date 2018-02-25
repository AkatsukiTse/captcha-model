from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.optimizers import SGD

import keras_model
import utils

HEIGHT = 70
WIDTH = 160
BATCH_SIZE = 64
CHAR_NUM = 62
CHAR_LEN = 4
FONT_SIZES = (40, 42, 44, 46, 48, 50)


def train():
    generator = utils.captcha_image_generator(batch_size=BATCH_SIZE, height=HEIGHT, width=WIDTH, font_sizes=FONT_SIZES)
    model = keras_model.resnet_model((HEIGHT, WIDTH, 3), CHAR_LEN, CHAR_NUM)
    model.compile(optimizer=SGD(lr=1e-2, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(
        generator,
        epochs=10, 
        steps_per_epoch=12800)

if __name__ == "__main__":
    train()