import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import utils.augmentation_functions as augmentation_utils
import cv2
import glob

DATASET_PATH = "CUB_200_2011"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
SEGMENTATION_PATH = os.path.join(DATASET_PATH, "segmentations")

N_CLASSES = 200

# mean = tf.constant([0.431776, 0.499619, 0.485914], dtype=tf.float32)
mean = tf.constant([0.43174747, 0.49959317, 0.48588511], dtype=tf.float32)
std = tf.constant([0.035313280220033036, 0.025757838145434496, 0.026814294497459704], dtype=tf.float32)


#@tf.function
def normalize_img(img):
    return tf.divide(img - mean, std, name="normalize")


#@tf.function
def denormalize_img(img):
    return tf.math.add(img * std, mean, name="denormalize")


def show_batch(image_batch, mask_batch, label_batch, show_mask=False):
    batch_size = image_batch.shape[0]

    columns = int(batch_size / 6)
    columns += 1 if (batch_size % 6) > 1 else 0

    plt.figure(figsize=(15, columns * 2.5))
    for n in range(batch_size):
        ax = plt.subplot(columns, 6, n + 1)
        plt.imshow(denormalize_img(image_batch[n]))
        plt.title("[%d]" % (np.argmax(label_batch[n]) + 1))
        plt.axis('off')

    plt.show()

    if show_mask:
        columns = int(batch_size / 6)
        columns += 1 if (batch_size % 6) > 1 else 0
        plt.figure(figsize=(15, columns * 2.5))
        for n in range(batch_size):
            ax = plt.subplot(columns, 6, n + 1)
            mask_rgb = cv2.cvtColor(mask_batch[n], cv2.COLOR_GRAY2RGB)
            plt.imshow(mask_rgb)
            plt.title("[%d]" % (np.argmax(label_batch[n]) + 1))
            plt.axis('off')

        plt.show()


#@tf.function
def pre_process_dataset(dataset, augmentation=False, with_mask=False, img_height=128, img_width=128):
    if augmentation:
        dataset = augmentation_utils.augment_dataset(dataset)

    if with_mask:
        dataset = dataset.map(get_segmented_image)

    # Normalization and resizing ______________
    dataset = dataset.map(lambda img, mask, label: (tf.image.resize(img, [img_height, img_width]),
                                                    tf.image.resize(mask, [img_height, img_width]),
                                                    label))
    dataset = dataset.map(lambda img, mask, label: (normalize_img(img), mask, label))

    # One Hot encoding in labels
    dataset = dataset.map(lambda img, mask, label: (img, mask, tf.one_hot(label, N_CLASSES)))

    return dataset


def load_dataset(shuffle=True):
    # Load image labels, training/test label and file path.
    train_labels = pd.read_csv(os.path.join(DATASET_PATH, "image_class_labels.txt"), header=None, sep=" ",
                               index_col=0, names=["class_label"])
    train_test = pd.read_csv(os.path.join(DATASET_PATH, "train_test_split.txt"), header=None, sep=" ",
                             index_col=0, names=["is_train"])
    images_paths = pd.read_csv(os.path.join(DATASET_PATH, "images.txt"), header=None, sep=" ", index_col=0,
                               names=["img_path"])
    # Generate path files for segmentation data
    segmentation_paths = images_paths.apply(lambda x: SEGMENTATION_PATH + os.path.sep + x.str.replace('.jpg', '.png'))
    segmentation_paths.rename(columns={"img_path": "seg_img_path"}, inplace=True, copy=False)
    # Generate path files for RGB data
    images_paths = images_paths.apply(lambda x: IMAGES_PATH + os.path.sep + x)

    # Combine dataset_df into single Pandas DataFrame
    # image ID is the index of the dataset_df DataFrame
    dataset_df = pd.concat((train_labels, train_test, images_paths, segmentation_paths), axis=1)

    train_df = dataset_df[dataset_df["is_train"] == 1]
    test_df = dataset_df[dataset_df["is_train"] == 0]

    # Generate validation set with samples of all classes
    val_df = pd.DataFrame([], columns=train_df.columns)
    for class_id in range(1, 201):
        class_samples = train_df[train_df["class_label"] == class_id].sample(n=3)
        val_df = pd.concat((val_df, class_samples), axis=0)

    train_df = train_df.drop(val_df.index)

    classes_df = pd.read_csv(os.path.join(DATASET_PATH, "classes.txt"), header=None, sep=" ", index_col=0,
                             names=["class"])

    if shuffle:
        train_df = train_df.sample(frac=1)
        val_df = val_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

    # Generate Dataset V2 input pipeline
    train_dataset = tf_dataset_from_dataframe(train_df)
    val_dataset = tf_dataset_from_dataframe(val_df)
    test_dataset = tf_dataset_from_dataframe(test_df)

    return train_dataset, val_dataset, test_dataset, classes_df


def tf_dataset_from_dataframe(data_df):
    img_paths = tf.convert_to_tensor(data_df['img_path'].values, dtype=tf.string)
    seg_img_paths = tf.convert_to_tensor(data_df['seg_img_path'].values, dtype=tf.string)
    labels = tf.convert_to_tensor(data_df['class_label'].values, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, seg_img_paths, labels))
    dataset = dataset.map(lambda img_path, mask_path, label: (get_img(img_path, channels=3, dtype=tf.float32),
                                                              get_mask(mask_path, channels=1, dtype=tf.float32),
                                                              label))
    return dataset


#@tf.function
def get_img(img_path, channels=3, dtype=tf.float32):
    img = tf.io.read_file(img_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=channels)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, dtype)
    return img


#@tf.function
def get_mask(img_path, channels=1, dtype=tf.float32):
    img = tf.io.read_file(img_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=channels)
    # Binarize mask
    img = tf.cast(tf.cast(img, tf.bool), tf.float32, name="binarize_img")
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, dtype)
    return img


#@tf.function
def get_segmented_image(img, mask, label):
    # mask = tf.cast(mask, dtype=tf.float32)
    img = tf.math.multiply(img, mask, name="mask_application")
    return img, mask, label


@DeprecationWarning
def load_tf_records(tf_records_dir="CALTECH_BIRDS_2011", with_info=False):
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

