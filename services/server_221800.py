"""Flask   2.2.3"""
from flask import Flask, request
import tensorflow as tf

app = Flask("Image classifier")
resnet = tf.keras.applications.ResNet101()
with open(
    "/Users/leonid/fa/pabd/pabd_cv/data/imgnet_cats_ru.txt", encoding="utf-8"
) as f:
    cats = f.readlines()

cats_en = [s.rstrip() for s in cats]


@app.route("/")
def home():
    """return string 'Home page' for / page"""
    return "Home page"


@app.route("/classify", methods=["POST", "GET"])
def classify():
    """docs"""
    data = request.get_data()
    img = tf.io.decode_jpeg(data)
    img_t = tf.convert_to_tensor(img, dtype=None, dtype_hint=None, name=None)
    img_t = tf.expand_dims(img_t, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out, direction="DESCENDING")[0][:3].numpy()
    out = ", ".join([cats_en[int(i)] for i in idxs])
    return out


if __name__ == "__main__":
    app.run(port=1800)  # номер зачетки
    input()
