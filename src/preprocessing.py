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

dataset_root = r'C:\Other Projects\open_cv_htr\dataset'

IMG_SIZE = 128
DATA_SIZE = 50000


# fixme change this method
def create_data():
    data = load_data()
    shuffle(data)  # randomly order data
    # data manipulation
    # image_data = data[0]
    # training_image_data = image_data[0:len(image_data) * 0.9]
    # testing_image_data = image_data[len(image_data) * 0.9: len(image_data) * 0.1]

    # label_data = data[1]
    # training_label_data = label_data[0:len(label_data) * 0.9]
    # testing_label_data = label_data[len(label_data) * 0.9: len(label_data) * 0.1]

    return train_test_split(data[0],
                            data[1],
                            train_size=0.9,
                            test_size=0.1,
                            random_state=123)


def load_data():
    print('Loading dataset')
    file_dir_names, class_names = load_annotation()
    print('Loading images')
    output = []

    # class_name_labels = {class_name: i for i, class_name in enumerate(class_names)}
    class_name_labels = {}
    for i in range(len(class_names)):
        class_name_labels[i] = class_names[i]
        pass
    images, labels = [], []

    broke_image_count = 0
    work_count = 0

    dataframe = dataset_root + r'\HK_dataset\simple_img'

    total_sum = 0
    for (dirpath, dirnames, filenames) in os.walk(dataframe):
        size_count = 0
        total_sum = len(filenames)
        for file in tqdm(filenames):
            label = class_name_labels[work_count]
            file = dirpath + r'\{}'.format(file)
            img_path = os.path.join(dataframe, file)
            if DATA_SIZE == -1 or size_count < DATA_SIZE:
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(label)
                except cv2.error as e:
                    broke_image_count += 1
                work_count += 1
            else:
                break
            pass
        pass

    print("Work image: {}".format(work_count))
    print("Broke image: {}".format(total_sum - work_count))

    print("Dataset: {}".format(len(images)))
    print("Labels: {}".format(len(labels)))

    return [np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.str)]
    pass


def load_annotation():
    print('Loading annotation from json')
    ann_path = dataset_root + r'\HK_dataset\simple_ann'

    dataframes = []
    file_names = []
    for (dir_path, dir_names, filenames) in os.walk(ann_path):
        size_count = 0
        for filename in tqdm(filenames):
            if DATA_SIZE == -1 or size_count < DATA_SIZE:
                f = open(dir_path + r'\{}'.format(filename), encoding="utf8")
                file_names.append(filename)
                dataframes.append(json.load(f)['description'])
            else:
                break
        break
    return file_names, dataframes
