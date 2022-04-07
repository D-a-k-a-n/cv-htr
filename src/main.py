import os
import shutil
from importlib import reload

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import ocr
import resnet as rs
import resnet_32x32 as rs_32
import preprocessing

import tensorflow as tf

root_path = r'/home/dakan/Documents/Computer Vision/cv-htr/dataset/'


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('Loss')
    ax[1].set_title("Accuracy")
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Test loss')
    ax[1].plot(history.history['accuracy'], label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Test Accuracy')

    #     ax.set_ylim(20)

    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')
    pass


def main():
    prep_data = preprocessing.create_data()
    training_image_data, testing_image_data, \
    training_label_data, testing_label_data = prep_data[2]

    le = prep_data[0]

    classWeight = prep_data[1]

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=False,
            fill_mode="nearest")

    #set config
    EPOCHS = 600
    INIT_LR = 1e-3
    BS = 128 #batch size

    opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model = rs_32.ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
    (64, 64, 128, 256), reg=0.0005)

    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"], run_eagerly=True)

    # H = model.fit(
    # aug.flow(training_image_data, training_label_data, batch_size=BS),
    #         validation_data=(testing_image_data, testing_label_data),
    #         steps_per_epoch=len(training_image_data) // BS,epochs=EPOCHS,
    #         class_weight=classWeight,
    #         verbose=1)

    # model = rs.ResNet(input_shape=(128, 128, 3), num_classes=8)
    # # opt = SGD(lr=0.01, momentum=0.7)
    # opt = keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1,
    #                              save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    # epochs = 50

    print(model.summary())

    print(training_image_data.shape)
    print(testing_image_data.shape)
    print(training_label_data.shape)
    print(testing_label_data.shape)

    # print(training_image_data.__len__())
    # print(testing_image_data.__len__())
    # print(training_label_data.__len__())
    # print(testing_label_data.__len__())

    # print(testing_image_data[0])
    # print(testing_label_data[0])

    # H = model.fit(x=training_image_data,
    #         validation_data=testing_image_data,
    #         steps_per_epoch=len(training_image_data) // BS,
    #         epochs=EPOCHS,
    #         class_weight=classWeight,
    #         verbose=1)

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "training_checks/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # batch_size = 32

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5*BS)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    # model.fit(train_images,
    #           train_labels,
    #           epochs=50,
    #           batch_size=batch_size,
    #       callbacks=[cp_callback],
    #       validation_data=(test_images, test_labels),
    #       verbose=0)


    model.fit(aug.flow(x=training_image_data, y=training_label_data, batch_size=BS),
        validation_data=(testing_image_data, testing_label_data),
        steps_per_epoch=len(training_image_data) // BS,
        epochs=EPOCHS,
        class_weight=classWeight,
        callbacks=[cp_callback],
        verbose=1)

    # history = model.fit(x=training_image_data,
    #                     y=training_label_data,
    #                     epochs=epochs,
    #                     validation_data=(testing_image_data, testing_label_data))
#
    # show_final_history(H)


#run
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()
