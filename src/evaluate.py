import tensorflow as tf
import os

model = tf.keras.models.load_model("models/my_model")


def resize(img, label):
    img_r = tf.image.resize(img, (180, 180))
    return img_r, label


if __name__ == "__main__":
    data_dir = "data/processed/train"
    test_datasets = tf.keras.utils.image_dataset_from_directory(data_dir)
    test_datasets = test_datasets.map(resize)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
    ]
    model.compile(metrics=metrics)
    history = model.evaluate(test_datasets)
    loss, accuracy, precision, recall = history
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
