import os
import numpy as np
import keras
from keras.preprocessing.image import list_pictures, load_img,img_to_array

def generate_image(source_directory_path, generate_directory_path, target_size, count_per_image):
    if not os.path.exists(generate_directory_path):
        os.mkdir(generate_directory_path)

    for picture in list_pictures(source_directory_path):
        img = load_img(picture,target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        generator = keras.preprocessing.image.ImageDataGenerator(   rescale= 1.0 / 255,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    horizontal_flip=True)

        g = generator.flow(x, batch_size=1, save_to_dir=generate_directory_path, save_prefix='img', save_format='bmp')
        for i in range(count_per_image):
            g.next()

if __name__ == '__main__':
    generate_image('./dataset/apple/', './dataset/apple_out/',(224,224), 8)