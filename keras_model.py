from keras.models import *
from keras.layers import *
from keras import backend as K

from resnet_component import *

def resnet_model(input_shape, char_len, char_num):
    input, x = ResnetBuilder.build(input_shape, 0, basic_block, [2, 2, 2, 2], return_flatten=True)
    x = Dropout(0.5)(x)
    output = [Dense(char_num, activation='softmax')(x) for i in range(char_len)]
    model = Model(inputs=input, outputs=output)
    return model





