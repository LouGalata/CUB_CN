import tensorflow as tf
import utils.birds_dataset_utils as dataset_utils

if __name__ == "__main__":
    train_dataset, test_dataset = dataset_utils.get_segmentation_dataset()
    train_dataset = dataset_utils.get_birds_tf_dataset(train_dataset.take(6), augmentation=True, with_mask=True)
    test_dataset = dataset_utils.get_birds_tf_dataset(test_dataset.take(6), with_mask=True)

    batch_size = 64 # 32 # 64
    # Show original image resized samples
    birds_tf_dataset = train_dataset.batch(batch_size)
    image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=True)

    birds_tf_dataset = test_dataset.batch(batch_size)
    image_batch, mask_batch, label_batch = next(iter(birds_tf_dataset))
    dataset_utils.show_batch(image_batch.numpy(), mask_batch.numpy(), label_batch.numpy(), show_mask=True)