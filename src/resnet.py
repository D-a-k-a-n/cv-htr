import os
import numpy as np
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Add, AveragePooling2D, \
    MaxPooling2D, ZeroPadding2D
from keras.optimizers import adadelta_v2 as adadelta
from keras.layers.core import Dense, Dropout
from tensorflow import keras
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

IMG_SIZE = 128  # resize image to this height and width
num_classes = 5  # different flower types
epochs = 30  # number of times model sees full data

lr = [0.5, 1, 2]  # learning rate options for hyperparam tuning
dropout_keep_rate = [0.9, 0.95, 1]  # dropout options for hyperparam tuning
batch_size = [16, 32, 64]  # batch size options for hyperparam tuning

image_dir = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset'  # location

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    # First Component of Main Path
    X = Conv2D(filters=3,
               kernel_size=(3, 3), strides=(1, 1),
               padding='same', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second Component of Main Path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third Component of Main Path
    X = Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
               padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_base_name = 'res_' + str(stage) + block + '_branch'
    bn_base_name = 'bn_' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    ### MAIN PATH ###
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(3, 3), strides=(s, s),
               padding='same', name=conv_base_name + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name + '2a')(X)
    X = Activation('relu')(X)

    # Second Component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_base_name + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name + '2b')(X)
    X = Activation('relu')(X)

    # Third Component of main path
    X = Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
               padding='same', name=conv_base_name + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name + '2c')(X)

    # Shortcut path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s),
                        padding='same', name=conv_base_name + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_base_name + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
    pass


def ResNet(input_shape, num_classes):
    model = Sequential()

    pre_model = keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        classes=num_classes,
    )

    model.add(pre_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(6000, activation='softmax'))

    return model
