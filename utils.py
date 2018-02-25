import random
import os

from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image

CHARACTERS = "".join([chr(i) for i in range(48, 58)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)])


def captcha_image_generator(
        batch_size=64,
        height=70,
        width=160,
        n_class=62,
        characters=CHARACTERS,
        char_num=4,
        font_sizes=(46)
        ):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(char_num)]
    generator = ImageCaptcha(width=width, height=height, font_sizes=font_sizes)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            image = generator.generate_image(random_str)
            X[i] = image
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def vetors_to_labels(y, chars=CHARACTERS):
    '''
    convert the one hot vetor to string labels
    :param y: one hot vetor with shape(char num, size of example, n_class)
    :param chars:
    :return:
    '''
    index = np.argmax(np.array(y), axis=2)
    char_num, m = index.shape
    return [''.join(chars[index[i][k]] for i in range(char_num)) for k in range(m)]


def evaluate_by_generator(generator, model, steps=10):
    '''
    evaluate the model by generator
    :param generator:
    :param model:
    :param steps:
    :return: accurary
    '''
    success = 0
    total = 0
    for i in range(steps):
        X, y = next(generator)
        predicts = vetors_to_labels(model.predict(X))
        labels = vetors_to_labels(y)
        m = X.shape[0]
        for j in range(m):
            if predicts[j].lower() == labels[j].lower():
                success += 1
            total += 1

    return success * 1.0 / total

def evaluate_by_files(folder, model, height, width, chars=CHARACTERS):
    success = 0
    labels = []
    X = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path).convert("RGB").resize((width, height))
        x = np.array(image, dtype=np.uint8)
        labels.append(file[:4])
        X.append(x)
    total = len(labels)
    X = np.array(X, dtype=np.uint8)
    predicts = model.predict(X)
    predict_labels = vetors_to_labels(predicts, chars)
    for i in range(total):
        if labels[i].lower() == predict_labels[i].lower():
            success += 1
    return success * 1.0 / total


