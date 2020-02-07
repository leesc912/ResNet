import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.image import (
    resize_image_with_crop_or_pad, random_crop, random_flip_left_right, per_image_standardization)
from math import ceil
import numpy as np

class CustomImageDataGenerator(tf.keras.utils.Sequence) :
    def __init__(self, X, y, image_shape, shuffle, batch_size, num_categories, data_augmentation) :
        super(CustomImageDataGenerator, self).__init__()

        self.X = X
        self.y = y
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.data_augmentation = data_augmentation
        self.num_data = len(self.X)

        self.on_epoch_end() # training을 시작하기 전, shuffle

    def __len__(self) :
        return ceil(self.num_data / self.batch_size)

    def __getitem__(self, idx) :
        return self._data_augmentation(
            self.indexList[idx * self.batch_size : (idx + 1) * self.batch_size]
        )

    def on_epoch_end(self) :
        self.indexList = np.arange(self.num_data)
        if self.shuffle :
            np.random.shuffle(self.indexList)

    def _data_augmentation(self, batch_index_list) :
        images = np.array([self.X[ID] for ID in batch_index_list], np.float32)
        labels = np.array([self.y[ID] for ID in batch_index_list], np.int32)

        # data augmentation
        if self.data_augmentation :
            # cifar : [40, 40, num_channels]
            images = resize_image_with_crop_or_pad(images, self.image_shape[0] + 8, self.image_shape[1] + 8)

            # cifar : [32, 32, num_channels]
            images = tf.map_fn(lambda image : random_crop(image, self.image_shape), images)
            images = tf.map_fn(lambda image : random_flip_left_right(image), images)

        images = tf.map_fn(lambda image : per_image_standardization(image), images)

        return images, labels

    def get_num_data(self) :
        return len(self.y)