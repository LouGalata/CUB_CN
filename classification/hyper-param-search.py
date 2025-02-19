import tensorflow as tf
import shutil
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorboard.plugins.hparams import api as hp

import numpy as np
import os
import utils.birds_dataset_utils as dataset_utils

# import utils.birds_dataset_utils as dataset_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_cnn_model(dropout, depth):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    N_CHANNELS = 3
    N_CLASSES = dataset_utils.N_CLASSES

    internal_cn_layers = []

    INITIAL_KERNELS = 32
    kernels_depth = [[INITIAL_KERNELS],
                     [INITIAL_KERNELS, INITIAL_KERNELS*2],
                     [INITIAL_KERNELS, INITIAL_KERNELS*2, INITIAL_KERNELS*4],
                     [INITIAL_KERNELS, INITIAL_KERNELS*2, INITIAL_KERNELS*4, INITIAL_KERNELS*4],
                     [INITIAL_KERNELS, INITIAL_KERNELS*2, INITIAL_KERNELS*2, INITIAL_KERNELS*4, INITIAL_KERNELS*4]]
    for n_kernel, d in zip(kernels_depth[depth - 1], range(1, depth + 1)):
        internal_cn_layers.append(tf.keras.layers.Conv2D(n_kernel, 3, padding='same', name="conv_%d" % d))
        internal_cn_layers.append(tf.keras.layers.BatchNormalization(axis=-1, name="b_norm_%d" % d))
        internal_cn_layers.append(tf.keras.layers.ReLU(name="RELU_%d" % d))
        internal_cn_layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

    model = tf.keras.models.Sequential(name="awsome_net", layers=
        # Input layer
        [tf.keras.layers.Conv2D(INITIAL_KERNELS, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_conv"),
         tf.keras.layers.BatchNormalization(axis=-1, name="input_batch_normalization"),
         tf.keras.layers.ReLU(name="input_RELU"),
         tf.keras.layers.MaxPooling2D((2, 2)),
         tf.keras.layers.Dropout(dropout)] +
        # Middle layers
        internal_cn_layers +
        # Output Layer
        [tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(dropout),
         tf.keras.layers.Dense(N_CLASSES, name="output_class_distribution")]
    )
    return model


def drop_ground_truth_segmentation(dataset):
    return dataset.map(lambda img, mask, label: (img, label))


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    AUGMENT = False
    MASK = True

    # Set hyper parameter search
    HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.000001, 0.001))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
    HP_DEPTH = hp.HParam('net_depth', hp.IntInterval(1, 5))
    HP_MASK = hp.HParam('maked_imgs', hp.Discrete([False, True]))

    hparams_log_dir = os.path.join("hyper-param-search", "logs")
    shutil.rmtree(hparams_log_dir, ignore_errors=True)
    hparams_writer = tf.summary.create_file_writer(hparams_log_dir)
    with hparams_writer.as_default():
        hp.hparams_config(
            hparams=[HP_LR, HP_DROPOUT, HP_DEPTH, HP_MASK],
            metrics=[
                hp.Metric('epoch_loss', group="train", display_name='epoch_loss'),
                hp.Metric('epoch_loss', group="validation", display_name='val_loss'),
                hp.Metric('epoch_categorical_accuracy', group="train", display_name='accuracy'),
                hp.Metric('epoch_categorical_accuracy', group="validation", display_name='val_accuracy'),
            ])

    epochs = 100
    print(HP_MASK.domain.values)
    for with_mask in [False, True]:
        train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=True)

        train_dataset = dataset_utils.pre_process_dataset(train_dataset, augmentation=True, with_mask=with_mask)
        test_dataset = dataset_utils.pre_process_dataset(test_dataset, with_mask=with_mask)

        train_dataset = drop_ground_truth_segmentation(train_dataset)
        test_dataset = drop_ground_truth_segmentation(test_dataset)

        BATCH_SIZE = 32
        train_dataset = train_dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        # Use for now part of test set
        val_dataset = test_dataset.take(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        # test_dataset = test_dataset.batch(BATCH_SIZE)
        # del(test_dataset)

        for lr in np.linspace(0.001, 0.00001, 4):
            for depth in range(2, 5 + 1):
                for dropout in np.linspace(0.3, 0, 2):

                    hparams = {
                        HP_DROPOUT: dropout,
                        HP_DEPTH: depth,
                        HP_LR: lr,
                        HP_MASK: with_mask,
                    }
                    # Run log dir
                    logdir = os.path.join(hparams_log_dir, "[32][A]lr=%.5f-D=%.2f-depth=%d-mask=%d" % (lr, dropout, depth, with_mask))

                    model = get_cnn_model(dropout=dropout, depth=depth)

                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                  metrics=[tf.keras.metrics.CategoricalAccuracy()]
                                  )
                    model.summary()

                    model.reset_states()
                    model.fit(train_dataset,
                              validation_data=val_dataset,
                              epochs=epochs,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                         tf.keras.callbacks.TensorBoard(logdir,
                                                                        update_freq='batch',
                                                                        write_graph=False,
                                                                        histogram_freq=5),
                                         tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                          patience=10),
                                         hp.KerasCallback(logdir, hparams, trial_id=logdir),
                                         tf.keras.callbacks.ModelCheckpoint(
                                             filepath=os.path.join(logdir, "checkpoints", "cp.ckpt"),
                                             save_best_only=True,
                                             monitor='val_loss',
                                             verbose=1)
                                         ]
                              )
