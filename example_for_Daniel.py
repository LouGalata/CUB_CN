import tensorflow as tf
import utils.birds_dataset_utils as dataset_utils

if __name__ == "__main__":

    print(tf.executing_eagerly())

    train_dataset, val_dataset, test_dataset, classes_df = dataset_utils.load_dataset(shuffle=True)

    # train_dataset = dataset_utils.pre_process_dataset(train_dataset.take(6), augmentation=False, with_mask=False)
    # test_dataset = dataset_utils.pre_process_dataset(test_dataset.take(6), with_mask=False)
    #
    # batch_size = 64 # 32 # 64
    # # Show original image resized samples
    # birds_tf_dataset = train_dataset.batch(batch_size)
    # image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    # dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=True)
    #
    # birds_tf_dataset = test_dataset.batch(batch_size)
    # image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    # dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=False)