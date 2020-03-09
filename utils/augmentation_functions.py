import scipy.ndimage as ndimage
import tensorflow as tf
import numpy as np


def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image

def tf_random_rotate_image(image, label):
  im_shape = image.shape
  [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
  image.set_shape(im_shape)
  return tf.clip_by_value(image, 0.0, 1.0), label


def crop_resize(img):
    BATCH_SIZE = tf.constant(1, dtype=tf.int32)
    NUM_BOXES = tf.constant(1, dtype=tf.int32)
    CROP_SIZE = (tf.constant(128, dtype=tf.int32), tf.constant(128, dtype=tf.int32))

    img = tf.expand_dims(img,0)
    left_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.05, maxval=0.2)
    right_corner = tf.random.uniform(shape=(NUM_BOXES, 2), minval=0.8, maxval=0.95)

    boxes = tf.concat((left_corner, right_corner), axis=1)
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    img = tf.image.crop_and_resize(img, boxes, box_indices, CROP_SIZE)[0]
    return img

def get_aspect_ratio(img, with_mask):
    target_height, target_width = 500, 500
    if with_mask:
        img_3d = img[:, :, :, 0]
        mask = img[:, :, :, 1]
        img_3d = tf.image.resize_with_pad(img_3d, target_height, target_width,
                                       method=tf.image.ResizeMethod.GAUSSIAN)
        mask = tf.image.resize_with_pad(mask, target_height, target_width,
                                       method=tf.image.ResizeMethod.GAUSSIAN)
        img = tf.stack([img_3d, mask], axis=3)
        return tf.clip_by_value(img, 0.0, 1.0)
    else:
        img = tf.image.resize_with_pad(img, target_height, target_width,
                                          method=tf.image.ResizeMethod.GAUSSIAN)
        return tf.clip_by_value(img, 0.0, 1.0)


def get_horizontal_flip(img, with_mask):
    if with_mask:
        img_3d = img[:, :, :, 0]
        mask = img[:, :, :, 1]
        img_3d = tf.image.flip_left_right(img_3d)
        mask = tf.image.flip_left_right(mask)
        return tf.stack([img_3d, mask], axis=3)
    else:
        return tf.image.flip_left_right(img)


def get_saturated_img(img, with_mask):
    if with_mask:
        img_3d = img[:, :, :, 0]
        mask = img[:, :, :, 1]
        img_3d = tf.image.random_saturation(img_3d, lower=1.5, upper=2.0, seed=103)
        return tf.stack([img_3d, mask], axis=3)
    else:
        return tf.image.random_saturation(img, lower=1.5, upper=2.0, seed=103)

def get_brightness_img(img, with_mask):
    if with_mask:
        img_3d = img[:, :, :, 0]
        mask = img[:, :, :, 1]
        img_3d = tf.clip_by_value(tf.image.random_brightness(img_3d, 0.3, seed=None), 0.0, 1.0)
        return tf.stack([img_3d, mask], axis=3)
    else:
        return tf.clip_by_value(tf.image.random_brightness(img, 0.3, seed=None), 0.0, 1.0)

def stuck_img_with_mask(img, mask):

    mask = tf.image.grayscale_to_rgb(mask)
    img = tf.stack([img, mask], axis=3)
    return img


def augment_dataset(dataset, with_mask=True):
    # Possible augmentations to perform __________________

    if with_mask:
        dataset = dataset.map(lambda img, mask, label: (stuck_img_with_mask(img, mask), label))
        # Keep the aspect ratio filling the gaps with padding

        dataset = dataset.map(lambda img, label: (get_aspect_ratio(img, with_mask),label))
        # Horizontal flip
        dataset = dataset.concatenate(dataset.map(lambda img, label: (get_horizontal_flip(img, with_mask), label)))
        # Change of saturation
        dataset = dataset.concatenate(dataset.map(lambda img, label:
                                                      (get_saturated_img(img, with_mask),label)))
        # Brightness
        dataset = dataset.concatenate(dataset.map(lambda img, label:
                                                  (get_brightness_img(img, with_mask), label)))
        dataset = dataset.map(lambda img, label: (get_segmented_image(img), label))
        # Rotation flip
        rotated_dataset = dataset.map(tf_random_rotate_image)
        # Croping
        cropped_dataset = rotated_dataset.map(lambda img, label: (crop_resize(img), label))
        dataset = dataset.concatenate(cropped_dataset)
    else:
        dataset = dataset.map(lambda img, mask, label: (img, label))
        dataset = dataset.map(lambda img, label: (get_aspect_ratio(img, with_mask),label))
        # Horizontal flip
        dataset = dataset.concatenate(dataset.map(lambda img, label: (get_horizontal_flip(img, with_mask), label)))
        # Change of saturation
        dataset = dataset.concatenate(dataset.map(lambda img, label:
                                                      (get_saturated_img(img, with_mask),label)))
        # Brightness
        dataset = dataset.concatenate(dataset.map(lambda img, label:
                                                  (get_brightness_img(img, with_mask), label)))
        # Rotation flip
        rotated_dataset = dataset.map(tf_random_rotate_image)
        # Croping
        cropped_dataset = rotated_dataset.map(lambda img, label: (crop_resize(img), label))
        dataset = dataset.concatenate(cropped_dataset)
    return dataset


def get_segmented_image(img):
    img_3d = img[:, :, :, 0]
    mask = img[:, :, :, 1]
    mask = tf.cast(mask, dtype=tf.float32)
    result = tf.math.multiply(img_3d, mask)
    return result
