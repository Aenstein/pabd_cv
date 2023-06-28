"""Flask   2.2.3"""
from flask import Flask, request
import tensorflow as tf

from utils import data_to_img, predict_imagenet

app = Flask("Image classifier")
resnet = tf.keras.applications.ResNet101()
with open("data/raw/imgnet_cats_ru.txt", encoding="utf-8") as f:
    cats = f.readlines()

categories_ru = [s.rstrip() for s in cats]


@app.route("/")
def home():
    """return string 'Home page' for / page"""
    return "Home page"


# @app.route("/classify/imagenet", methods=["POST", "GET"])
# def classify():
#     """docs"""
#     data = request.get_data()
#     img = tf.io.decode_jpeg(data)
#     img_t = tf.convert_to_tensor(img, dtype=None, dtype_hint=None, name=None)
#     img_t = tf.expand_dims(img_t, axis=0)
#     img_t = tf.image.resize(img_t, (224, 224))
#     out = resnet(img_t)
#     idxs = tf.argsort(out, direction="DESCENDING")[0][:3].numpy()
#     out = ", ".join([categories_ru[int(i)] for i in idxs])
#     return out


@app.route("/classify/binary", methods=["POST"])
def classify_binary():
    """binary"""
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (180, 180))
    model = tf.keras.models.load_model("models/my_model")
    out = model(img_t)
    print(out)
    dog_probability = out.numpy()[0, 0]
    print(dog_probability)
    idx = dog_probability > 0.5
    return ("Cat", "Dog")[idx]


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    input()
