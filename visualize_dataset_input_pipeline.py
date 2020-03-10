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

    train_dataset, test_dataset = dataset_utils.get_segmentation_dataset()
    train_dataset = dataset_utils.get_birds_tf_dataset(train_dataset.take(6), augmentation=True, with_mask=True)
    test_dataset = dataset_utils.get_birds_tf_dataset(test_dataset.take(6), with_mask=True)


    batch_size = 64 # 32 # 64
    # Show original image resized samples
    birds_tf_dataset = train_dataset.batch(batch_size)
    image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy())

    birds_tf_dataset = test_dataset.batch(batch_size)
    image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy())



    # # Construct a tf.data.Dataset
    # dataset, dataset_info = tfds.load(name="caltech_birds2011", download=False,
    #                                   data_dir="tf_CUB_200_2011", split="false", with_info=True)


    # # Build your input pipeline
    # dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # for features in dataset.take(1):
    #     image, label, mask = features["image"], features["label"], features['segmentation_mask']
    #
    #     break
