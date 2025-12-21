import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np


def prepare_dataset(data_dir, img_height, img_width, batch_size):

    # 1. Wczytanie danych
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode=None,
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    # 2. Normalizacja (0-255 -> 0-1)
    normalization_layer = layers.Rescaling(1. / 255)
    ds = ds.map(lambda x: normalization_layer(x))

    # 3. Augmentacja
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Funkcja mapująca dla Autoenkodera:
    def augment_wrapper(image):
        aug_image = data_augmentation(image, training=True)
        return aug_image, aug_image

    ds = ds.map(augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def main():
  config = {
    'data_dir': 'dataset_images',
    'img_height': 128,
    'img_width': 128,
    'batch_size': 16,
    'latent_dim': 2,
    'epochs': 50
  }
  train_ds = prepare_dataset(
      config['data_dir'],
      config['img_height'],
      config['img_width'],
      config['batch_size']
  )

  if train_ds is None:
      return