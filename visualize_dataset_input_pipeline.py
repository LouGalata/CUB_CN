from __future__ import absolute_import, division, print_function, unicode_literals
# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow_datasets as tfds

# %load_ext tensorboard
# %load_ext autoreload
import scipy
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pandas as pd
import numpy as np
import os
import datetime

from pathlib import Path

import utils.birds_dataset_utils as dataset_utils

if __name__ == "__main__":
    tf.compat.v1.set_random_seed(1634)

    current_dir = os.getcwd()
    print("Current directory is ", current_dir)

    train_df, test_df = dataset_utils.get_segmentation_dataset()

    #
    # train_df, val_df, test_df, classes_df = dataset_utils.load_dataset()
    # pd.DataFrame.hist(train_df, bins=200, column="class_label")
    # plt.title("Training set class distribution")
    # plt.show()
    # pd.DataFrame.hist(test_df, bins=200, column="class_label")
    # plt.title("Testing set class distribution")
    # plt.show()
    # print("Training dataset_df contains %d images_paths" % train_df.shape[0])
    # print("Validation dataset_df contains %d images_paths" % val_df.shape[0])
    # print(train_df.sample(5))
    # print("Test dataset_df contains %d images_paths" % test_df.shape[0])
    # # print(test_df.sample(5))
    # print("Existing classes ")
    # print(classes_df.head(5))
    #
    birds_tf_dataset = dataset_utils.get_birds_tf_dataset(train_df.take(6), augmentation=True, with_mask=True)

    #
    batch_size = 64 # 32 # 64
    # Show original image resized samples
    birds_tf_dataset = birds_tf_dataset.batch(batch_size)
    image_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), label_batch.numpy())

    birds_tf_test_dataset = dataset_utils.get_birds_tf_dataset(test_df.take(6), with_mask=True)
    birds_tf_test_dataset = birds_tf_test_dataset.batch(batch_size)

    image_batch, label_batch = next(iter(birds_tf_test_dataset))
    dataset_utils.show_batch(image_batch.numpy(), label_batch.numpy())


    # # Construct a tf.data.Dataset
    # dataset, dataset_info = tfds.load(name="caltech_birds2011", download=False,
    #                                   data_dir="tf_CUB_200_2011", split="false", with_info=True)


    # # Build your input pipeline
    # dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # for features in dataset.take(1):
    #     image, label, mask = features["image"], features["label"], features['segmentation_mask']
    #
    #     break
