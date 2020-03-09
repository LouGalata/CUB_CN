# from __future__ import absolute_import, division, print_function, unicode_literals
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
    tf.test.gpu_device_name()

    tf.compat.v1.set_random_seed(1634)

    current_dir = os.getcwd()
    print("Current directory is ", current_dir)

    train_dataset, test_dataset = dataset_utils.get_segmentation_dataset()
    train_dataset = dataset_utils.get_birds_tf_dataset(train_dataset.take(6), augmentation=True, with_mask=True)
    test_dataset = dataset_utils.get_birds_tf_dataset(test_dataset.take(6), augmentation=False, with_mask=True)

    #
    batch_size = 64 # 32 # 64
    # Show original image resized samples
    birds_tf_dataset = train_dataset.batch(batch_size)
    image_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), label_batch.numpy())

    birds_tf_dataset = test_dataset.batch(batch_size)
    image_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), label_batch.numpy())

