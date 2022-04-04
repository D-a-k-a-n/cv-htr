"""
Data presented in 5 different folders, one for each class ~800 images per folderA
function creates two .npy files containing image data and labels with roughly even distribution of flower types in each
train = 90%, test = 10%
combines into single list containing 2 arrays. [0] = image data, [1] = one hot encoded class label
"""
import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle
from sklearn import preprocessing

dataset_root = r'/home/dakan/Documents/Computer Vision/cv-htr/dataset/'

IMG_SIZE = 128
DATA_SIZE = 1000

# fixme change this method
def create_data():
    data = load_data()
    # shuffle(data)  # randomly order data

    X = data[0]
    y = data[1]

    #lebel encoder
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    #output
    return train_test_split(X,
                            y,
                            train_size=0.9,
                            test_size=0.1,
                            random_state=123)


def load_data():
    print('Loading dataset')
    # first row of list {'name': '827_015_005.jpg', 'description': 'сату'}
    ann_list = load_annotation() #load dicts in list with img path and label

    print('Loading images')

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
            img = cv2.imread(img_link, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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
