import os
import numpy
import cv2

picture_directory_path = './image_cutout/'
dataset_directory_path = './dataset/'


size = (128,128)
directory_names = [("haze",1)] #名前がいけてないけど、ディレクトリ名と魚に対応する値のタプルのリスト
train_csv_file_name = "train.csv"

with open(os.path.join(dataset_directory_path, train_csv_file_name), 'w') as f:
    for directory_name in directory_names:
        if not os.path.exists(os.path.join(dataset_directory_path,directory_name[0])):
            os.mkdir(os.path.join(dataset_directory_path,directory_name[0]))

        file_names = os.listdir(os.path.join(picture_directory_path, directory_name[0]))

        for file_name in file_names:
            resize_file_path = os.path.join(picture_directory_path,directory_name[0],file_name)
            img = cv2.imread(resize_file_path)
            new_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(dataset_directory_path,directory_name[0],file_name),new_img)
            f.write(f"{resize_file_path},{directory_name[1]}"+'\n')