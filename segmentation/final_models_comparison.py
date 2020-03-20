import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from utils import birds_dataset_utils as dataset_utils
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix, accuracy_score

DATASET_PATH = "CUB_200_2011"


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


def display_predictions(display_list, show=True, max_rows=4):
    imgs, masks, predictions = display_list
    batch_size = imgs.shape[0] if len(imgs.shape) == 4 else 1
    num_rows = max_rows if batch_size > max_rows else batch_size
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    img_size = 2.5
    n_cols = 6
    fig = plt.figure(figsize=(img_size * n_cols, img_size * num_rows))
    for row in range(num_rows):
        plt.subplot(num_rows, n_cols, (row * n_cols) + 1)
        plt.imshow(dataset_utils.denormalize_img(imgs[row]))
        plt.axis('off')

        if row == 0:
            plt.title("Input")

        plt.subplot(num_rows, n_cols, (row * n_cols) + 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[row]),
                   cmap='gray')
        plt.axis('off')

        if row == 0:
            plt.title("Ground Truth")

        plt.subplot(num_rows, n_cols, (row * n_cols) + 3)
        pred_mask = predictions[row]
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask),
                   cmap='gray')
        plt.axis('off')

        if row == 0:
            plt.title("Net output")


        plt.subplot(num_rows, n_cols, (row * n_cols) + 4)
        pred_mask = tf.reduce_all(predictions[row] > 0.1, axis=-1, keepdims=True)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask),
                   cmap='gray')
        plt.axis('off')

        if row == 0:
            plt.title("Binarize th=0.10")


        plt.subplot(num_rows, n_cols, (row * n_cols) + 5)
        pred_mask = tf.reduce_all(predictions[row] > 0.25, axis=-1, keepdims=True)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask),
                   cmap='gray')
        plt.axis('off')

        if row == 0:
            plt.title("Binarize th=0.25")


        plt.subplot(num_rows, n_cols, (row * n_cols) + 6)
        pred_mask = tf.reduce_all(predictions[row] > 0.5, axis=-1, keepdims=True)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask),
                   cmap='gray')
        if row == 0:
            plt.title("Binarize th=0.50")
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


if __name__ == "__main__":
    # Load the predictions labels as dataframes
    # Load image labels, training/test label and file path.
    train_labels = pd.read_csv(os.path.join(DATASET_PATH, "image_class_labels.txt"), header=None, sep=" ",
                               index_col=0, names=["class_label"])
    train_test = pd.read_csv(os.path.join(DATASET_PATH, "train_test_split.txt"), header=None, sep=" ",
                             index_col=0, names=["is_train"])

    dataset_df = pd.concat((train_labels, train_test), axis=1)
    y_train = dataset_df[dataset_df["is_train"] == 1]['class_label'].values
    y_test = dataset_df[dataset_df["is_train"] == 0]['class_label'].values

    # __________________________________________________________________________________________
    # Get dataset in TF format
    train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=True)

    test_dataset = dataset_utils.pre_process_dataset(test_dataset, with_mask=False)

    # Drop class label as we are appoaching a segmentation task
    test_dataset = test_dataset.map(lambda img, mask, label: (img, mask))
    test_dataset = test_dataset.batch(16)
    image_batch, mask_batch = next(iter(test_dataset))

    path = "segmentation\seg-hyper-param-search\logs"

    for model_log_dir in os.listdir(path=path):
        if not os.path.isdir(os.path.join(path, model_log_dir)):
            continue

        pos_weight = int(model_log_dir.split('pw=')[1])
        model_ckpt_path = os.path.join(path, model_log_dir, "checkpoints", "cp.ckpt")
        # Avoid compilation since custom loss function was used
        seg_model = tf.keras.models.load_model(model_ckpt_path, compile=False)
        # Compile the model with the same objects
        seg_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                          loss=WeightedBinaryCrossentropy(pos_weight=pos_weight))
        print("Model " + model_log_dir)

        # Save model architecture to log folder
        model_img_path = os.path.join(path, model_log_dir, "model_architecture.png")
        tf.keras.utils.plot_model(model=seg_model, to_file=model_img_path, show_shapes=True)

        # Save model H5 and .Jason architecture
        seg_model.save(os.path.join(path, model_log_dir, "model.h5"))
        json_string = seg_model.to_json()
        with open(os.path.join(path, model_log_dir, 'model.json'), 'w') as outfile:
            json.dump(json_string, outfile)

        predictions = seg_model.predict(image_batch)

        fig = display_predictions([image_batch, mask_batch, predictions], show=False, max_rows=10)
        plt.savefig(os.path.join("segmentation", "seg-hyper-param-search", model_log_dir + ".png"), format='png')
        plt.show()

        # break
