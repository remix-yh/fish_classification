import os
import csv
import numpy as np
import pandas as pd
import PIL
import tensorflow
from keras.applications.vgg16 import VGG16
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


epochs = 100
batch_size = 8
img_size = (64,64)
category_size = 3
model_file_path = './model/model_weight.h5'

def get_data_list(data_list, target_size, category_size):
    X = []
    y = []

    for item in data_list:
        for picture in list_pictures(item['directory_path']):
            img = img_to_array(load_img(picture, target_size=target_size))
            X.append(img)
            y.append(item['correct_value'])

    # arrayに変換
    X = np.asarray(X)
    y = np.asarray(y)

    # 画素値を0から1の範囲に変換
    X = X.astype('float32')
    X = X / 255.0

    # クラスの形式を変換
    y = np_utils.to_categorical(y, category_size)

    return (X, y)

def load_data():

    #教師データの読込
    X_train = []
    y_train = []

    train_data_list = [
        {   "directory_path" : './dataset/haze/train/',     "correct_value" : 0 },
        {   "directory_path" : './dataset/kasago/train/',   "correct_value" : 1 },
        {   "directory_path" : './dataset/aji/train/',     "correct_value" : 2 }
    ]

    (X_train,y_train) = get_data_list(train_data_list, img_size, category_size)

    #テストデータの読み込み
    X_test = []
    y_test = []

    test_data_list = [
        {   "directory_path" : './dataset/haze/test/',     "correct_value" : 0 },
        {   "directory_path" : './dataset/kasago/test/',   "correct_value" : 1 },
        {   "directory_path" : './dataset/aji/test/',     "correct_value" : 2 }
    ]
    (X_test,y_test) = get_data_list(test_data_list, img_size, category_size)

    return X_train, X_test, y_train, y_test

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape = (64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(category_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy']) 

    return model

def create_vgg16_model():
    vgg16 = VGG16(include_top=False, input_shape=(224,224,3))
    model = Sequential(vgg16.layers)
    for layer in model.layers[:15]:
        layer.trainable = False
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(category_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=1e-4,momentum=0.9),
                metrics=['accuracy']) 
    
    return model

def load_weight(model, file_path):
    if not os.path.exists(file_path):
        return 
    model.load_weights(file_path)

def fit(model, x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test))
    write_log(history)
    model.save(model_file_path)

def initialize(filePath):
    pred_model = create_model()
    load_weight(pred_model, filePath)
    return pred_model

def predict(model, img):
    img = img.resize((64,64))
    img_list = []
    img_list.append(img_to_array(img))
    img_np_list = np.asarray(img_list)
    img_np_list = img_np_list.astype('float32')
    img_np_list = img_np_list / 255.0
    score = model.predict(img_np_list, 1)

    return score

def write_log(history):
    with open('./log/history.csv', 'w', newline='') as f : 
        w = csv.writer(f)
        w.writerow(['acc','loss', 'val_acc', 'val_loss'])
        for i in range(len(history.epoch)):
            w.writerow([    history.history['acc'][i],
                            history.history['loss'][i],
                            history.history['val_acc'][i],
                            history.history['val_loss'][i]
                            ])

if __name__ == '__main__':

    #教師用データ、テスト用データ読込
    X_train, X_test, y_train, y_test = load_data()

    #モデル生成
    model = create_model()
    
    #重み読込
    load_weight(model,model_file_path)

    #学習
    fit(model, X_train, X_test, y_train, y_test)