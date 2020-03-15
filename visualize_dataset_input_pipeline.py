# from __future__ import absolute_import, division, print_function, unicode_literals
# %tensorflow_version 2.x
# %load_ext tensorboard
# %load_ext autoreload
import os

import tensorflow as tf

import utils.birds_dataset_utils as dataset_utils

if __name__ == "__main__":
    tf.test.gpu_device_name()

    tf.compat.v1.set_random_seed(1634)

    current_dir = os.getcwd()
    print("Current directory is ", current_dir)

    train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=True)

    train_dataset = dataset_utils.pre_process_dataset(train_dataset.take(6), augmentation=True, with_mask=False)
    test_dataset = dataset_utils.pre_process_dataset(test_dataset.take(6), with_mask=False)

    # Get datasets ready and optimize input pipeline by
    # 1. Shuffling
    # 2. Batching
    # 3. Prefetching
    BATCH_SIZE = 64 # 32 # 64
    train_dataset = train_dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Show original image resized samples
    image_batch, mask_batch, label_batch = next(iter(train_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=True)
    #
    birds_tf_dataset = test_dataset.batch(BATCH_SIZE)
    image_batch, mask_batch, label_batch = next(iter(test_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=False)

