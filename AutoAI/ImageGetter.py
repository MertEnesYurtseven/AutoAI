import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
img_height = 28
img_width = 28
batch_size = 2
main_folder=r"MNIST Dataset JPG format/MNIST - JPG - training"


def ImageGetterFromSubClassFolders(main_folder):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        main_folder,
        labels="inferred",
        label_mode="int",  # categorical, binary
        # class_names=['0', '1', '2', '3', ...]
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training",
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        main_folder,
        labels="inferred",
        label_mode="int",  # categorical, binary
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="validation",
    )
    return ds_train,ds_validation





