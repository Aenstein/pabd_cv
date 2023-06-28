import tensorflow as tf


def data_to_img(data):
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    return tf.image.resize(img_t, (180, 180))


def predict_imagenet(img, model):
    out = model(img)
    return tf.argsort(out, direction='DESCENDING')[0][:3].numpy()