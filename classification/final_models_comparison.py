import json
import os

import pandas as pd
import tensorflow as tf

from utils import birds_dataset_utils as dataset_utils

DATASET_PATH = "CUB_200_2011"


def drop_ground_truth_segmentation(dataset):
    return dataset.map(lambda img, mask, label: (img, label), num_parallel_calls=tf.data.experimental.AUTOTUNE)


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
    train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=False)
    train_dataset = dataset_utils.pre_process_dataset(train_dataset, augmentation=True, with_mask=False)
    test_dataset = dataset_utils.pre_process_dataset(test_dataset, with_mask=False)
    train_dataset = drop_ground_truth_segmentation(train_dataset)
    test_dataset = drop_ground_truth_segmentation(test_dataset)

    path = "classification\logs-best-runs\logs"

    for model_log_dir in os.listdir(path=path):
        if not os.path.isdir(os.path.join(path, model_log_dir)):
            continue

        model_ckpt_path = os.path.join(path, model_log_dir, "checkpoints", "cp.ckpt")
        best_model = tf.keras.models.load_model(model_ckpt_path)

        # Save model architecture to log folder
        model_img_path = os.path.join(path, model_log_dir, "model_architecture.png")
        tf.keras.utils.plot_model(model=best_model, to_file=model_img_path, show_shapes=True)

        # Save model H5 and .Jason architecture
        best_model.save(os.path.join(path, model_log_dir, "model.h5"))
        json_string = best_model.to_json()
        with open(os.path.join(path, model_log_dir, 'model.json'), 'w') as outfile:
            json.dump(json_string, outfile)

        # # Test the model on test set
        # predictions = best_model.predict(test_dataset.batch(16))
        #
        # pred_distributions = tf.nn.softmax(predictions)
        # y_test_pred = tf.argmax(pred_distributions, axis=-1)
        #
        # acc = accuracy_score(y_test, y_test_pred)
        # cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred, normalize='true')
        # plt.figure(figsize=(10, 8))
        # plt.title("Caltech Birds Dataset Classification\n%s\n Acc: %.4f  Test-set-samples: %d N_Classes:200" %
        #           (model_log_dir, acc, len(y_test)))
        # sns.heatmap(cm, vmin=0., vmax=1.)
        # plt.tight_layout()
        # plt.savefig(os.path.join("classification", "logs-best-runs", model_log_dir + ".png"), format='png')
        # plt.show()