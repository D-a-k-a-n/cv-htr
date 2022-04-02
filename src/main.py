import os
import shutil
from importlib import reload

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow import keras

import ocr
import resnet as rs
import preprocessing

root_path = r'C:\Other Projects\open_cv_htr\dataset'


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
    batch_size = 128

    training_image_data, testing_image_data, \
    training_label_data, testing_label_data = preprocessing.create_data()

    # rs.resnet(training_image_data, training_label_data,
    #           testing_image_data, testing_label_data)

    model = rs.ResNet(input_shape=(128, 128, 3), classes=8)
    # opt = SGD(lr=0.01, momentum=0.7)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    epochs = 5

    print(model.summary())

    print(training_image_data.shape)
    print(testing_image_data.shape)
    print(training_label_data.shape)
    print(testing_label_data.shape)

    # print(testing_image_data[0])
    print(testing_label_data[0])

    history = model.fit(x=training_image_data,
                        y=training_label_data,
                        epochs=epochs,
                        validation_data=(testing_image_data, testing_label_data))

    show_final_history(history)
    pass


def test_data():
    img_path = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\img'
    ann_path = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\ann'

    ann_original = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\ann'
    ann_target = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\simple_ann'

    img_original = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\img'
    img_target = r'C:\Other Projects\open_cv_htr\dataset\HK_dataset\simple_img'

    for (dir_path, dir_names, filenames) in os.walk(ann_path):
        count = 0
        for filename in filenames:
            if count < 10000:
                shutil.copyfile(os.path.join(ann_original, filename), os.path.join(ann_target + filename))
                count += 1
            else:
                break

    for (dir_path, dir_names, filenames) in os.walk(img_path):
        count = 0
        for filename in filenames:
            if count < 10000:
                shutil.copyfile(os.path.join(img_original, filename), os.path.join(img_target, filename))
                count += 1
            else:
                break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
