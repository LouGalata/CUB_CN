import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow_addons as tfa



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


def get_birds_tf_dataset(data, augmentation=False, aspect_ratio=False):
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

    if augmentation:
        dataset = augment_dataset(dataset, aspect_ratio)

    # Normalization and resizing ______________
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]), label))
    dataset = dataset.map(lambda img, label: (normalize_img(img), label))

    return dataset


@tf.function
def rotate_tf(image):
    if image.shape.__len__() == 4:
        random_angles = tf.random.uniform(shape=(tf.shape(image)[0],), minval=-np.pi / 4, maxval=np.pi / 4)

    # Outputs random values from a uniform distribution, where minval = pi/4 and maxval = pi/4
    if image.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape=(), minval=-np.pi / 6, maxval=np.pi / 6)

    return tfa.image.rotate(image, random_angles)


def crop_resize(img):
    BATCH_SIZE = 1
    NUM_BOXES = 1
    CROP_SIZE = (128, 128)

    img = tf.expand_dims(img, 0)
    left_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.05, maxval=0.2)
    right_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.8, maxval=0.95)

    boxes = tf.concat((left_corner, right_corner), axis=1)
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    output = tf.image.crop_and_resize(img, boxes, box_indices, CROP_SIZE)
    return output[0]

def augment_dataset(dataset, aspect_ratio = False):
    # Possible augmentations to perform __________________

    # Keep the aspect ratio filling the gaps with padding
    if aspect_ratio:
        target_height, target_width = 200, 200
        dataset = dataset.map(lambda img, label: (tf.clip_by_value(tf.image.resize_with_pad(
                                                      img,  target_height, target_width, method=tf.image.ResizeMethod.GAUSSIAN, antialias=True)
                                                      , 0.0, 1.0), label))
    # Horizontal flip
    dataset = dataset.concatenate(dataset.map(lambda img, label: (tf.image.flip_left_right(img), label)))

    # Change of saturation
    dataset = dataset.concatenate(dataset.map(lambda img, label:
                                                  (tf.image.random_saturation(img, lower=1.5, upper=2.0, seed=103),label)))
    # Brightness
    dataset = dataset.concatenate(dataset.map(lambda img, label:
                                              (tf.clip_by_value(tf.image.random_brightness(img, 0.3, seed=None),0.0, 1.0), label)))

    # Rotation flip
    rotated_dataset = dataset.map(lambda img, label: (rotate_tf(img), label))

    # Croping
    cropped_dataset = rotated_dataset.map(lambda img, label: (crop_resize(img), label))

    dataset = dataset.concatenate(cropped_dataset)
    return dataset


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

    # Generate validation set with samples of all classes
    val_df = pd.DataFrame([], columns=train_df.columns)
    for class_id in range(1, 201):
        class_samples = train_df[train_df["class_label"] == class_id].sample(n=4)
        val_df = pd.concat((val_df, class_samples), axis=0)

    train_df = train_df.drop(val_df.index)

    classes_df = pd.read_csv(DATASET_PATH + "classes.txt", header=None, sep=" ", index_col=0,
                             names=["class"])

    return train_df, val_df, test_df, classes_df
