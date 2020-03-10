import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf


def random_rotate_image_mask(image, mask):
    random = np.random.uniform(-30, 30)
    image = ndimage.rotate(image, random, reshape=False)
    mask = ndimage.rotate(mask, random, reshape=False)
    return image, mask


def random_rotate_image(image):
    random = np.random.uniform(-30, 30)
    image = ndimage.rotate(image, random, reshape=False)
    return image


def tf_random_rotate_image(image, mask, label):
    im_shape = image.shape
    mask_shape = mask.shape
    [image, mask, ] = tf.py_function(random_rotate_image_mask, [image, mask], [tf.float32, tf.float32])
    image.set_shape(im_shape)
    mask.set_shape(mask_shape)
    return tf.clip_by_value(image, 0.0, 1.0), tf.clip_by_value(mask, 0.0, 1.0), label


def crop_resize(img, mask, label):
    BATCH_SIZE = tf.constant(1, dtype=tf.int32)
    NUM_BOXES = tf.constant(1, dtype=tf.int32)
    CROP_SIZE = (tf.constant(128, dtype=tf.int32), tf.constant(128, dtype=tf.int32))

    img = tf.expand_dims(img, 0)
    mask = tf.expand_dims(mask, 0)
    left_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.05, maxval=0.2)
    right_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.8, maxval=0.95)

    boxes = tf.concat((left_corner, right_corner), axis=1)
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    img = tf.image.crop_and_resize(img, boxes, box_indices, CROP_SIZE)[0]
    mask = tf.image.crop_and_resize(mask, boxes, box_indices, CROP_SIZE)[0]
    return img, mask, label


def get_masked_horizontal_flip(img, mask, label):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask, label


def get_saturated_img(img, mask, label):
    img = tf.image.random_saturation(img, lower=1.5, upper=2.0, seed=103)
    return img, mask, label


def get_brightness_img(img, mask, label):
    img = tf.clip_by_value(tf.image.random_brightness(img, 0.3, seed=None), 0.0, 1.0)
    return img, mask, label


def stuck_img_with_mask(img, mask):
    mask = tf.image.grayscale_to_rgb(mask)
    img = tf.stack([img, mask], axis=3)
    return img


# Possible augmentations to perform __________________
def augment_dataset(dataset):
    # Horizontal flip
    dataset = dataset.concatenate(dataset.map(get_masked_horizontal_flip))

    # Change of saturation
    saturated_dataset = dataset.map(get_saturated_img)
    dataset = dataset.concatenate(saturated_dataset)

    # Brightness
    brightness_dataset = dataset.map(get_brightness_img)
    dataset = brightness_dataset.concatenate(brightness_dataset)

    # Rotation flip
    rotated_dataset = dataset.map(tf_random_rotate_image)
    # Croping
    cropped_dataset = rotated_dataset.map(crop_resize)
    dataset = dataset.concatenate(cropped_dataset)
    return dataset
