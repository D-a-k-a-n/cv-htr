import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.preprocessing import LabelBinarizer

dataset_root = r'/home/dakan/Documents/Computer Vision/cv-htr/dataset/'

IMG_SIZE = 32
DATA_SIZE = 10000
BS = 128 #batch size

# fixme change this method
def create_data():
    data = load_data()
    # shuffle(data)  # randomly order data

    X = data[0]
    y = data[1]

    aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=False,
            fill_mode="nearest")

    #lebel encoder
    le = preprocessing.LabelBinarizer()
    # le.fit(y)
    y = le.fit_transform(y)

    # account for skew in the labeled data

    classTotals = y.sum(axis=0)
    classWeight = {}

    # loop over all classes and calculate the class weight
    for i in range(0, len(classTotals)):
      classWeight[i] = classTotals.max() / classTotals[i]




    training_image_data, testing_image_data, \
    training_label_data, testing_label_data = train_test_split(X,y,
                                                               train_size=0.8,
                                                               test_size=0.2,
                                                               random_state=123)

    # training_image_data = aug.flow(x=training_image_data, y=training_label_data, batch_size=BS)
    # testing_image_data = aug.flow(x=testing_image_data, y=testing_label_data, batch_size=BS)

    # exit()

    #output
    return le, classWeight, (training_image_data, testing_image_data,
                            training_label_data, testing_label_data)

def load_data():
    print('Loading dataset')
    # first row of list {'name': '827_015_005.jpg', 'description': 'сату'}
    ann_list = load_annotation() #load dicts in list with img path and label

    print('Loading images')
    #
    # train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
    # # test_datagen = ImageDataGenerator()
    #
    # train_generator = train_datagen.flow_from_directory(f"{dataset_root}/HK_dataset/img",target_size=(224, 224),batch_size=32,shuffle=True,class_mode='categorical')
    # # test_generator = test_datagen.flow_from_directory("/content/gdrive/My Drive/datasets/test",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')

    if DATA_SIZE != -1:
        ann_list = ann_list[:DATA_SIZE]
    # output = []

    images, labels = [], []

    broke_image_count = 0

    img_path = dataset_root + 'HK_dataset/img'
    total_sum = len(ann_list)
    for ann_dict in tqdm(ann_list):
        label = ann_dict['description']
        img_link = f"{img_path}/{ann_dict['name']}"
        try:
            img = cv2.imread(img_link, 0)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.reshape(32, 32, 1)
            images.append(img)
            labels.append(label)
        except cv2.error as e:
            broke_image_count += 1

    print("Work image: {}".format(total_sum - broke_image_count))
    print("Broke image: {}".format(broke_image_count))

    print("Dataset: {}".format(len(images)))
    print("Labels: {}".format(len(labels)))

    return [np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.str)]


def load_annotation():
    ann_path = dataset_root + 'HK_dataset/ann'
    ann_json_list = os.listdir(ann_path)

    ann = []
    for file_name in tqdm(ann_json_list):
        path = ann_path + '/' + file_name
        with open(path) as file:
            ann.append(json.load(file))

    return ann
