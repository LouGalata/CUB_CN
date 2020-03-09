import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import utils.augmentation_functions as augmentation_utils


DATASET_PATH = "CUB_200_2011/"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")

IMG_HEIGHT = tf.constant(128, dtype=tf.int32)
IMG_WIDTH = tf.constant(128, dtype=tf.int32)
N_CHANNELS = tf.constant(3, dtype=tf.int32)
N_CLASSES = tf.constant(200, dtype=tf.int32)

# mean = tf.constant([0.431776, 0.499619, 0.485914], dtype=tf.float32)
mean = tf.constant([0.43174747, 0.49959317, 0.48588511], dtype=tf.float32)
std = tf.constant([0.035313280220033036, 0.025757838145434496, 0.026814294497459704], dtype=tf.float32)


def normalize_img(img):
    return tf.divide(img - mean, std)


def denormalize_img(img):
    return tf.math.add(img * std, mean)


def show_batch(image_batch, label_batch):
    batch_size = image_batch.shape[0]

    columns = int(batch_size / 6)
    columns += 1 if (batch_size % 6) > 1 else 0

    plt.figure(figsize=(15, columns * 2.5))
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


def get_birds_tf_dataset(dataset, augmentation=False, with_mask=False):
    if augmentation:  # TRAIN DATASET
        dataset = augmentation_utils.augment_dataset(dataset, with_mask)
    elif with_mask: # TEST DATASET WITH MASK
        dataset = dataset.map(lambda img, mask, label: (augmentation_utils.stuck_img_with_mask(img, mask), label))
        dataset = dataset.map(lambda img, label: (augmentation_utils.get_aspect_ratio(img, with_mask), label))
        dataset = dataset.map(lambda img, label: (augmentation_utils.get_segmented_image(img), label))
    else: # TEST DATASET WITH NO MASK
        dataset = dataset.map(lambda img, mask, label: (img, label))
        dataset = dataset.map(lambda img, label: (augmentation_utils.get_aspect_ratio(img, with_mask), label))


    # Normalization and resizing ______________
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]), label))
    dataset = dataset.map(lambda img, label: (normalize_img(img), label))

    #One Hot encoding in labels
    dataset = dataset.map(lambda img, label: (img, tf.one_hot(label, N_CLASSES)))

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


def get_segmentation_dataset(tf_records_dir="CALTECH_BIRDS_2011", with_info=False):
    dataset, info = tfds.load(name="caltech_birds2011",
                              data_dir=tf_records_dir,
                              split=None,
                              shuffle_files=False, with_info=True)
    training_dataset = dataset['train']
    test_dataset = dataset['test']

    # Extract (image, mask, label) tuples
    training_dataset = training_dataset.map(lambda x:
                                            (tf.image.convert_image_dtype(x['image'], tf.float32),
                                             tf.cast(tf.cast(x['segmentation_mask'], tf.bool), tf.float32),
                                             tf.cast(x['label'], tf.int32)))
    test_dataset = test_dataset.map(lambda x:
                                    (tf.image.convert_image_dtype(x['image'], tf.float32),
                                     tf.cast(tf.cast(x['segmentation_mask'], tf.bool), tf.float32),
                                     tf.cast(x['label'], tf.int32)))

    if with_info:
        return training_dataset, test_dataset, info
    else:
        return training_dataset, test_dataset


def get_segmented_tst_image_with_mask(img, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    result = tf.math.multiply(img, mask)
    return result