import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorboard.plugins.hparams import api as hp

import numpy as np
import os
import utils.birds_dataset_utils as birds_dataset_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_cnn_model(dropout, depth):
    IMG_HEIGHT = birds_dataset_utils.IMG_HEIGHT
    IMG_WIDTH = birds_dataset_utils.IMG_WIDTH
    N_CHANNELS = birds_dataset_utils.N_CHANNELS
    N_CLASSES = birds_dataset_utils.N_CLASSES

    internal_cn_layers = []
    kernels_depth = [[16], [16, 32], [16, 32, 64]]
    for n_kernel, d in zip(kernels_depth[depth - 1], range(1, depth + 1)):
        internal_cn_layers.append(tf.keras.layers.Conv2D(n_kernel, 3, padding='same', name="conv_%d" % d))
        internal_cn_layers.append(tf.keras.layers.BatchNormalization(axis=-1, name="b_norm_%d" % d))
        internal_cn_layers.append(tf.keras.layers.ReLU(name="RELU_%d" % d))
        internal_cn_layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

    model = tf.keras.models.Sequential(name="awsome_net", layers=
        # Input layer
        [tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_conv"),
         tf.keras.layers.BatchNormalization(axis=-1, name="input_batch_normalization"),
         tf.keras.layers.ReLU(name="input_RELU"),
         tf.keras.layers.MaxPooling2D((2, 2)),
         tf.keras.layers.Dropout(dropout)] +
        # Middle layers
        internal_cn_layers +
        # Output Layer
        [tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(dropout),
         tf.keras.layers.Dense(N_CLASSES, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation='softmax',
                               name="output_class_distribution")]
    )
    return model


if __name__ == '__main__':
    # Load the dataset_df
    train_df, val_df, test_df, classes_df = birds_dataset_utils.load_dataset()

    training_dataset = birds_dataset_utils.get_birds_tf_dataset(train_df.sample(100),
                                                                rand_saturation=True,
                                                                horizontal_flip=True)
    validation_dataset = birds_dataset_utils.get_birds_tf_dataset(val_df,
                                                                  rand_saturation=False,
                                                                  horizontal_flip=False)
    batch_size = 32
    training_dataset = training_dataset.batch(32)
    validation_dataset = validation_dataset.batch(32)

    # Set hyper parameter search
    HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.00001, 0.001))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([.2, .3, .4, .5]))
    HP_DEPTH = hp.HParam('net_depth', hp.Discrete([1, 2, 3]))

    hparams_log_dir = os.path.join("hyper-param-search", "logs")
    hparams_writer = tf.summary.create_file_writer(hparams_log_dir)
    with hparams_writer.as_default():
        hp.hparams_config(
            hparams=[HP_LR, HP_DROPOUT],
            metrics=[
                hp.Metric('epoch_loss', display_name='epoch_loss'),
                hp.Metric('epoch_val_loss', display_name='epoch_val_loss'),
                hp.Metric('epoch_categorical_accuracy', display_name='epoch_categorical_accuracy'),
            ])

    epochs = 100
    batch_size = 32
    for dropout in HP_DROPOUT.domain.values:
        for depth in HP_DEPTH.domain.values:
            for lr in np.linspace(HP_LR.domain.max_value, HP_LR.domain.min_value, 4):
                hparams = {
                    HP_DROPOUT: dropout,
                    HP_DEPTH: depth,
                    HP_LR: lr,
                }
                # Run log dir
                logdir = os.path.join(hparams_log_dir, "lr=%.5f-D=%.2f-depth=%d" % (lr, dropout, depth))

                model = get_cnn_model(dropout=dropout, depth=depth)

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=[tf.keras.metrics.CategoricalAccuracy()]
                              )
                model.summary()

                model.reset_states()
                model.fit(training_dataset,
                          validation_data=validation_dataset,
                          epochs=3,
                          callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                     tf.keras.callbacks.TensorBoard(logdir,
                                                                    update_freq='batch',
                                                                    # write_images=True,
                                                                    histogram_freq=1),
                                     tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=10),
                                     hp.KerasCallback(logdir + '/train', hparams, trial_id=logdir)
                                     ]
                          )
