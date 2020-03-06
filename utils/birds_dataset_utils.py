import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
DATASET_PATH = "CUB_200_2011/"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")

IMG_HEIGHT = 128
IMG_WIDTH = 128
N_CHANNELS = 3
N_CLASSES = 200

# mean = tf.constant([0.431776, 0.499619, 0.485914], dtype=tf.float32)
mean = tf.constant([0.43174747, 0.49959317, 0.48588511], dtype=tf.float32)
std = tf.constant([0.035313280220033036, 0.025757838145434496, 0.026814294497459704], dtype=tf.float32)


def normalize_img(img):
    return tf.divide(img - mean, std)


def denormalize_img(img):
    return tf.math.add(img * std, mean)


def show_batch(image_batch, label_batch):
    batch_size = image_batch.shape[0]

    columns = int(batch_size/6)
    columns += 1 if (batch_size % 6) > 1 else 0

    plt.figure(figsize=(15, columns*2.5))
    for n in range(batch_size):
        ax = plt.subplot(columns, 6, n + 1)
        plt.imshow(denormalize_img(image_batch[n]))
        plt.title("[%d]" % (np.argmax(label_batch[n]) + 1))
        plt.axis('on')

    plt.show()


def get_img(img_path):
    img = tf.io.read_file(img_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img
    # return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def get_birds_tf_dataset(data, rand_saturation=True, horizontal_flip=True):
    """
    This method builds an input pipeline for the birds dataset images applying optional data augmentation
    :param data: Dataframe with the class labels and image paths
    :param rand_saturation:
    :param horizontal_flip:
    :return: Unbatched dataset of loaded images and their correspondent one_hot_enconded labels
    """
    imagepaths = tf.convert_to_tensor(data['img_path'].values, dtype=tf.string)
    labels = tf.convert_to_tensor(data['class_label'].values, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))
    # Load images and set one hot encoded labels
    dataset = dataset.map(lambda img_path, label: (get_img(img_path),
                                                   tf.one_hot(label-1, N_CLASSES)))

    # Possible augmentations to perform __________________
    # Change of saturation
    if rand_saturation:
        sat_dataset = dataset.map(lambda img, label: (tf.image.random_saturation(img, lower=0.2, upper=1.8, seed=103),
                                                      label))
        dataset = dataset.concatenate(sat_dataset)
    # Horizontal flip
    if horizontal_flip:
        dataset = dataset.concatenate(dataset.map(lambda img, label: (tf.image.flip_left_right(img), label)))

    # Normalization and resizing ______________
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]), label))
    dataset = dataset.map(lambda img, label: (normalize_img(img), label))

    return dataset


def augment_dataset(dataset):
    horizontal_flip = dataset.map(lambda img, label: (tf.image.flip_left_right(img), label))
    return horizontal_flip


def load_dataset():
    # Load image labels, training/test label and file path.
    train_labels = pd.read_csv(DATASET_PATH + "image_class_labels.txt", header=None, sep=" ",
                               index_col=0, names=["class_label"])
    train_test = pd.read_csv(DATASET_PATH + "train_test_split.txt", header=None, sep=" ",
                             index_col=0, names=["is_train"])
    images_paths = pd.read_csv(DATASET_PATH + "images.txt", header=None, sep=" ", index_col=0,
                               names=["img_path"])
    # Complete relative path
    images_paths = images_paths.apply(lambda x: IMAGES_PATH + "/" + x)

    # Combine dataset_df into single Pandas DataFrame
    # image ID is the index of the dataset_df DataFrame
    dataset_df = pd.concat((train_labels, train_test, images_paths), axis=1)

    train_df = dataset_df[dataset_df["is_train"] == 1]
    test_df = dataset_df[dataset_df["is_train"] == 0]

    classes_df = pd.read_csv(DATASET_PATH + "classes.txt", header=None, sep=" ", index_col=0,
                             names=["class"])

    return train_df, test_df, classes_df
