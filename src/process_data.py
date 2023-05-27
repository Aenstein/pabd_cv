"""Preprocess data to use it as tf dataset.
Take images from source dir and put in out dir subfolders: Cat and Dog"""
import os.path
import shutil
import tensorflow as tf
import click
import PIL


# in_dir = 'data\\raw\\kaggle'
# out_dir = 'data\\processed\\PetImages'
# n_img = 20


@click.command()
@click.option("-i", "--in_dir", default="data/raw/kaggle")
@click.option("-o", "--out_dir", default="data/processed/train")
@click.option("-n", "--n_img", default=20)
@click.option("-s", "--img_size", default=180)
# @click.option("-f", "--filter_corrupted", default=False)
def process_data(in_dir, out_dir, n_img, img_size):
    """All"""
    make_out_dir(out_dir)
    copy_cats_imgs(in_dir, out_dir, n_img, img_size)
    copy_dogs_imgs(in_dir, out_dir, n_img, img_size)
    # filter_corrupted(out_dir)


def make_out_dir(out_dir):
    """Make dir"""
    if os.path.exists(out_dir):
        os.remove(out_dir)
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, "cat"))
    os.mkdir(os.path.join(out_dir, "dog"))


def copy_cats_imgs(in_dir, out_dir, n_img, img_size):
    """Make cats images for training"""
    all_imgs = os.listdir(in_dir)
    cat_imgs = [img for img in all_imgs if img.startswith("cat")]

    for cat_img in cat_imgs[:n_img]:
        in_ing_path = os.path.join(in_dir, cat_img)
        img = PIL.Image.open(in_ing_path)
        img_r = img.resize((img_size, img_size))
        out_img_path = os.path.join(out_dir, "cat", cat_img)
        img_r.save(out_img_path)


def copy_dogs_imgs(in_dir, out_dir, n_img, img_size):
    """Make dogs images for training"""
    all_imgs = os.listdir(in_dir)
    dog_imgs = [img for img in all_imgs if img.startswith("dog")]
    
    for dog_img in dog_imgs[:n_img]:
        in_ing_path = os.path.join(in_dir, dog_img)
        img = PIL.Image.open(in_ing_path)
        img_r = img.resize((img_size, img_size))
        out_img_path = os.path.join(out_dir, "dog", dog_img)
        img_r.save(out_img_path)


# def filter_corrupted(out_dir):
#     """..."""
#     num_skipped = 0
#     for folder_name in ("Cat", "Dog"):
#         folder_path = os.path.join(out_dir, folder_name)
#         for fname in os.listdir(folder_path):
#             fpath = os.path.join(folder_path, fname)
#             try:
#                 fobj = open(fpath, "rb")
#                 is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#             finally:
#                 fobj.close()

#             if not is_jfif:
#                 num_skipped += 1
#                 # Delete corrupted image
#                 os.remove(fpath)


if __name__ == "__main__":
    process_data()
