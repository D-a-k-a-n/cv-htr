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



# def create_image_label(train_data, test_data):
#     """ deepnote
#     Images are now:
#     IMG_SIZE*IMG_SIZE*RGB attached to one hot class label type and ordered randomly
#     Data is comprised of a list containing: [0]: image data, [1]: class label
#     """
#     # create image (arrays) and label (lists) for use in models
#     X_train = np.array([item[0] for item in train_data]).reshape(-1, IMG_SIZE, 3)
#     Y_train = np.array([item[1] for item in train_data])
#     x_valid = np.array([item[0] for item in test_data]).reshape(-1, IMG_SIZE, 3)
#     y_valid = np.array([item[1] for item in test_data])
#
#     X_train = X_train / 255  # normalising
#     x_valid = x_valid / 255  # normalising
#
#     # create results file
#     with open('results_file.csv', 'w') as f:
#         f.write('lr,keep rate,batch size,val_accuracy,epoch\n')
#
#     resnet(X_train, Y_train, x_valid, y_valid)


# def resnet(X_train, Y_train, x_valid, y_valid):
#     # ---------- MODELLING AND TESTING ----------
#     # for loops for grid search of hyperparameter options
#     for i in lr:
#         for j in dropout_keep_rate:
#             for k in batch_size:
#                 # to save well performing models
#                 MODEL_NAME = 'handwriting-{}-{}-{}-{}-{}.model' \
#                     .format('lr' + str(i), 'dr' + str(j), 'bs' + str(k), '5-layer', 'resnet-basic')
#
#                 model = Sequential()
#                 # layer 1
#                 model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # layer 2
#                 model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # layer 3
#                 model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # layer 4
#                 model.add(Conv2D(256, (3, 3), input_shape=(16, 16, 3), activation='relu'))
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # layer 5
#                 model.add(Conv2D(512, (3, 3), input_shape=(8, 8, 3), activation='relu'))
#                 # fully connected
#                 model.add(Flatten())
#                 model.add(Dense(128, activation='relu'))
#                 model.add(Dropout(j))
#                 model.add(Dense(5, activation='softmax'))
#
#                 adadelt = adadelta.Adadelta(lr=i, decay=1e-6)
#                 model.compile(loss='binary_crossentropy', optimizer=adadelt, metrics=['accuracy'])
#
#                 out = model.fit(X_train,
#                                 Y_train,
#                                 batch_size=k,
#                                 epochs=epochs,
#                                 verbose=1,
#                                 validation_data=(x_valid, y_valid))
#
#                 # show model testing parameters
#                 print(out.params)
#
#                 max_val_acc = max(out.history['val_acc'])
#                 max_ep = [a + 1 for a, b in enumerate(out.history['val_acc']) if b == max_val_acc]
#                 with open('results_file.csv', 'a') as f:
#                     f.write('{},{},{},{},{}\n'.format(i,
#                                                       j,
#                                                       k,
#                                                       max_val_acc,
#                                                       max_ep[0]))
#
#                 # un-comment below to save model
#                 # model.save(MODEL_NAME)


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
