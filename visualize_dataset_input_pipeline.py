from __future__ import absolute_import, division, print_function, unicode_literals
# %tensorflow_version 2.x
import tensorflow as tf

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

    train_df, test_df, classes_df = dataset_utils.load_dataset()
    print("Training dataset_df contains %d images_paths" % train_df.shape[0])
    print(train_df.sample(5))
    print("Test dataset_df contains %d images_paths" % test_df.shape[0])
    print(test_df.sample(5))
    print("Existing classes ")
    print(classes_df.head(5))

    birds_tf_dataset = dataset_utils.get_birds_tf_dataset(train_df.head(6),
                                                          rand_saturation=True,
                                                          horizontal_flip=True)

    batch_size = 32  # 32 # 64
    # Show original image resized samples
    birds_tf_dataset = birds_tf_dataset.batch(batch_size)
    image_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), label_batch.numpy())
