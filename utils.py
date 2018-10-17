import os
import cv2
import pickle
import numpy as np


image_size = 256
image_channel = 3
label_size = 150
unknown_label_size = 50
train_data_path = 'dataset/train_images'

processed_image_data_dir = 'processed_picture/image_data.p'
processed_string_label_data_dir = 'processed_picture/string_label_data.p'
processed_numeral_label_data_dir = 'processed_picture/numeral_label_data.p'
processed_parameter_dir = 'processed_picture/parameter.p'
processed_one_hot_label_data_dir = 'processed_picture/one_hot_label_data.p'




def load_data(path=train_data_path):
    '''
    将训练的图片数据集和label载入程序
    其中label list采用one hot编码
    :param path: 数据集文件夹路径
    '''
    image_data = []
    label_data = []
    i = 0
    for dir in os.listdir(path=train_data_path):
        curr_path = path + '/' + dir
        i += 1
        if i <= label_size+1: # 只读取150个labels，剩余的50用于devise model的测试
            if os.path.isdir(curr_path):
                for data in os.listdir(curr_path):
                    image = cv2.imread(curr_path + '/' + data)
                    image = cv2.resize(image, (image_size, image_size))  # 图像和原图像具有相同的内容，只是大小和原图像不一样而已
                    image_data.append(image)
                    label_data.append(dir.split('.')[1])

    # 对label进行onehot编码
    label_set = list(set(label_data)) #len(label_set)=150
    label2int = {val: i for i, val in enumerate(label_set)}
    int2label = {i: val for i, val in enumerate(label_set)}

    numeral_labels = [label2int[i] for i in label_data]
    one_hot_labels = []
    for label in numeral_labels:
        vector = [0] * len(label_set)
        vector[label] = 1
        one_hot_labels.append(vector)
    print('data processing finished')

    print(len(label_data))
    print(len(numeral_labels))
    print(len(one_hot_labels))

    pickle.dump(image_data, open(processed_image_data_dir,'wb'))
    pickle.dump([label2int, int2label], open(processed_parameter_dir, 'wb'))
    pickle.dump(numeral_labels, open(processed_numeral_label_data_dir, 'wb'))
    pickle.dump(one_hot_labels, open(processed_one_hot_label_data_dir, 'wb'))
    pickle.dump(label_data, open(processed_string_label_data_dir, 'wb') )
    return image_data, one_hot_labels, label2int, int2label

def get_image_data():
    images = pickle.load(open(processed_image_data_dir, 'rb'))
    return images

def get_one_hot_label_data():
    one_hot_label = pickle.load(open(processed_one_hot_label_data_dir, 'rb'))
    return one_hot_label

def get_numeral_label_data():
    numeral_label = pickle.load(open(processed_numeral_label_data_dir, 'rb'))
    return numeral_label

def get_parameter():
    label2int, int2label = pickle.load(open(processed_parameter_dir, 'rb'))
    return label2int, int2label







if __name__ =='__main__':
    load_data()