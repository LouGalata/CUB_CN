from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import shutil
from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt
import os
import io
import utils.birds_dataset_utils as dataset_utils

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class LogSegmentationPredCallback(tf.keras.callbacks.Callback):

    def __init__(self, logdir, dataset, img_title="model_pred", rate=1):
        super(LogSegmentationPredCallback, self).__init__()
        self.file_writer = tf.summary.create_file_writer(os.path.join(logdir, "pred_imgs"))
        self.dataset = dataset
        self.img_title = img_title
        self.rate = rate

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 != 0:
            return

        for img, mask in self.dataset.take(1):
            pred_mask = self.model.predict(img)
            display_list = [img, mask, pred_mask]
            figure = self._display_predictions(display_list, show=False, max_rows=4)
            with self.file_writer.as_default():
                tf.summary.image(self.img_title, self._plot_to_image(figure), step=epoch)
                print("Logging predictions...")

    def _create_mask(self, pred_mask):
        if pred_mask.shape[2] > 1:
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
        # else:
        # pred_mask = tf.cast(pred_mask, dtype=tf.int8)
        return pred_mask

    def _display_predictions(self, display_list, show=True, max_rows=4):
        imgs, masks, predictions = display_list
        batch_size = imgs.shape[0] if len(imgs.shape) == 4 else 1
        num_rows = max_rows if batch_size > max_rows else batch_size
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        img_size = 2.5
        fig = plt.figure(figsize=(img_size * 3, img_size * num_rows))

        for row in range(num_rows):
            plt.subplot(num_rows, 3, (row * 3) + 1)
            plt.imshow(dataset_utils.denormalize_img(imgs[row]))

            plt.subplot(num_rows, 3, (row * 3) + 2)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[row]),
                       cmap='gray')

            plt.subplot(num_rows, 3, (row * 3) + 3)
            pred_mask = self._create_mask(predictions[row])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask),
                       cmap='gray')
            plt.axis('on')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig

    def _plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):

    def __init__(self, pos_weight, name='WeightedBinaryCrossentropy'):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        # For numerical stability clip predictions to log stable numbers
        y_pred = tf.keras.backend.clip(y_pred,
                                       tf.keras.backend.epsilon(),
                                       1 - tf.keras.backend.epsilon())
        # Compute weighted binary cross entropy
        wbce = y_true * -tf.math.log(y_pred) * self.pos_weight + (1 - y_true) * -tf.math.log(1 - y_pred)
        # Reduce by mean
        return tf.reduce_mean(wbce)


def downsample(filters, size, stride=2, apply_batchnorm=True, use_bias=False,
               name=None):
    if not name is None:
        name = "%s_conv2d_k%d_s%d" % (name, size, stride)
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
                                      padding='same', use_bias=use_bias))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    if stride == 1:
        result.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))

    return result


def upsample(filters, size, stride=2, apply_dropout=False, use_bias=False, name=None):
    if not name is None:
        name = "%s_conv2d_trans_k%d_s%d%s" % (name, size, stride,
                                              "_D" if apply_dropout else "")
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                               padding='same', use_bias=use_bias))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def SegmentationModel(kernel_size=3, strides=1, depth=8, dropout=True,
                      skip_connections=True, output_channels=2):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    down_stack = [
        downsample(64, size=kernel_size, stride=strides, apply_batchnorm=False, name="1"),  # (bs, 128, 128, 64)
        downsample(128, size=kernel_size, stride=strides, name="2"),  # (bs, 64, 64, 128)
        downsample(256, size=kernel_size, stride=strides, name="3"),  # (bs, 32, 32, 256)
        downsample(512, size=kernel_size, stride=strides, name="4"),  # (bs, 16, 16, 512)
        downsample(512, size=kernel_size, stride=strides, name="5"),  # (bs, 8, 8, 512)
        downsample(512, size=kernel_size, stride=strides, name="6"),  # (bs, 4, 4, 512)
        downsample(512, size=kernel_size, stride=strides, name="7"),  # (bs, 2, 2, 512)
        downsample(512, size=kernel_size, stride=strides, name="8"),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, size=kernel_size, name="8", apply_dropout=dropout),  # (bs, 2, 2, 1024)
        upsample(512, size=kernel_size, name="7", apply_dropout=dropout),  # (bs, 4, 4, 1024)
        upsample(512, size=kernel_size, name="6", apply_dropout=dropout),  # (bs, 8, 8, 1024)
        upsample(512, size=kernel_size, name="5", apply_dropout=dropout),  # (bs, 16, 16, 1024)
        upsample(256, size=kernel_size, name="4", ),  # (bs, 32, 32, 512)
        upsample(128, size=kernel_size, name="3", ),  # (bs, 64, 64, 256)
        upsample(64, size=kernel_size, name="2", ),  # (bs, 128, 128, 128)
    ]

    # Limit model by the provided depth
    down_stack = down_stack[:depth]
    up_stack = up_stack[-depth + 1:]

    # Leave last layer without activation as a loss with parameter `from_logits`
    # will be used.
    last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size,
                                           strides=2,
                                           activation='sigmoid',
                                           padding='same')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        if skip_connections:
            x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=True)

    train_dataset = dataset_utils.pre_process_dataset(train_dataset, augmentation=True, with_mask=False)
    test_dataset = dataset_utils.pre_process_dataset(test_dataset, with_mask=False)
    val_dataset = dataset_utils.pre_process_dataset(val_dataset, with_mask=False)

    # Drop class label as we are appoaching a segmentation task
    train_dataset = train_dataset.map(lambda img, mask, label: (img, mask))
    val_dataset = val_dataset.map(lambda img, mask, label: (img, mask))
    test_dataset = test_dataset.map(lambda img, mask, label: (img, mask))

    # Get datasets ready and optimize input pipeline by
    # 1. Shuffling (only training set)
    # 2. Batching
    # 3. Prefetching
    BATCH_SIZE = 10  # 32 # 64
    train_dataset = train_dataset.take(1000).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Define metrics to watch
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    # Set hyper parameter search
    HP_DEPTH = hp.HParam('net_depth', hp.IntInterval(3, 8))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([False, True]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 4]))
    HP_DOWN_STRIDE = hp.HParam('down_stride', hp.Discrete([1, 2]))
    HP_POS_CLASS_WEIGHT = hp.HParam('pos_class_weight', hp.Discrete([1., 1.5, 2., 10., 50.]))

    hparams_log_dir = os.path.join("segmentation", "logs")
    shutil.rmtree(hparams_log_dir, ignore_errors=True)
    hparams_writer = tf.summary.create_file_writer(hparams_log_dir)
    with hparams_writer.as_default():
        hp.hparams_config(
            hparams=[HP_KERNEL_SIZE, HP_DROPOUT, HP_DEPTH, HP_DOWN_STRIDE],
            metrics=[
                hp.Metric('epoch_loss', group="train", display_name='epoch_loss'),
                hp.Metric('epoch_loss', group="validation", display_name='val_loss'),
                hp.Metric('auc', group="train", display_name='auc'),
                hp.Metric('auc', group="validation", display_name='val_auc'),
                hp.Metric('precision', group="train", display_name='precision'),
                hp.Metric('precision', group="validation", display_name='precision_val'),
                hp.Metric('recall', group="train", display_name='recall'),
                hp.Metric('recall', group="validation", display_name='recall_val'),
            ])

    EPOCHS = 2

    for depth in range(HP_DEPTH.domain.min_value, HP_DEPTH.domain.max_value + 1):
        for kernel_size in HP_KERNEL_SIZE.domain.values:
            for down_stride in HP_DOWN_STRIDE.domain.values:
                # for dropout in HP_DROPOUT.domain.values:
                for pos_weight in HP_POS_CLASS_WEIGHT.domain.values:
                    dropout = False
                    hparams = {
                        HP_DEPTH: depth,
                        HP_DROPOUT: dropout,
                        HP_KERNEL_SIZE: kernel_size,
                        HP_DOWN_STRIDE: down_stride,
                        HP_POS_CLASS_WEIGHT: pos_weight,
                    }
                    # Run log dir
                    logdir = os.path.join(hparams_log_dir, "depth=%d-k=%d-s=%d-pw=%d%s" %
                                          (depth, kernel_size, down_stride, pos_weight,
                                           "_Drop" if dropout else ""))

                    if os.path.exists(logdir):
                        continue

                    model = SegmentationModel(kernel_size=kernel_size, strides=down_stride,
                                              depth=depth, dropout=dropout,
                                              skip_connections=True, output_channels=1)

                    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                  loss=WeightedBinaryCrossentropy(pos_weight=pos_weight),
                                  metrics=METRICS)

                    model.summary()
                    model.reset_states()

                    model.fit(train_dataset,
                              validation_data=val_dataset,
                              epochs=EPOCHS,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                         tf.keras.callbacks.TensorBoard(logdir,
                                                                        update_freq='batch',
                                                                        write_graph=False,
                                                                        histogram_freq=5),
                                         tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                          patience=6),
                                         hp.KerasCallback(logdir, hparams, trial_id=logdir),
                                         # LogSegmentationPredCallback(logdir, test_dataset,
                                         #                             img_title="test_set",
                                         #                             rate=2),
                                         tf.keras.callbacks.ModelCheckpoint(
                                             filepath=os.path.join(logdir, "checkpoints", "cp.ckpt"),
                                             save_best_only=True,
                                             monitor='val_loss',
                                             verbose=1)
                                         ]
                              )
                    tf.keras.backend.clear_session()

