import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = 'dataset_images'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
LATENT_DIM = 2  # Wymiar przestrzeni ukrytej
EPOCHS = 50  # Liczba epok treningu


def prepare_dataset(data_dir, img_height, img_width, batch_size):

    # 1. Wczytanie danych
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode=None,  # Autoenkoder nie potrzebuje etykiet
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

    def augment_wrapper(image):
        aug_image = data_augmentation(image, training=True)
        return aug_image, aug_image

    ds = ds.map(augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def build_autoencoder(img_height, img_width, latent_dim):
    # buduje enkoder
    inputs = Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)

    shape_before_flatten = x.shape[1:]

    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, name="latent_space")(x)

    encoder = models.Model(inputs, latent_vector, name="Encoder")

    # dekoder
    latent_inputs = Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)

    decoder_outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = models.Model(latent_inputs, decoder_outputs, name="Decoder")

    # autoenkoder
    autoencoder_input = Input(shape=(img_height, img_width, 3))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    autoencoder = models.Model(autoencoder_input, decoded, name="Autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def visualize_reconstruction(model, dataset, n=5):
    # Wizualizacja
    # Pobranie jednej paczki danych
    batch = next(iter(dataset))
    images, _ = batch
    images = images[:n]

    reconstructed = model.predict(images)

    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Wejście
        plt.imshow(images[i])
        plt.title("Wejście")
        plt.axis("off")

        # Wyjście
        plt.imshow(reconstructed[i])
        plt.title("Rekonstrukcja")
        plt.axis("off")

    plt.show()

def visualize_latent_space(encoder, dataset, num_batches=5):

    all_images = []
    # Pobieramy kilka batchy, żeby mieć więcej punktow na wykresie
    for img_batch, _ in dataset.take(num_batches):
        all_images.append(img_batch)

    if not all_images:
        return

    all_images = np.concatenate(all_images)
    latent_coords = encoder.predict(all_images)

    plt.figure(figsize=(8, 8))
    plt.scatter(latent_coords[:, 0], latent_coords[:, 1], c='blue', alpha=0.6)
    plt.title("Latent Space (2D)")
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.grid(True)
    plt.show()

def main():

    train_ds = prepare_dataset(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    if train_ds is None:
        return  # Przerywamy jeśli brak danych

    # 2. Budowa modelu
    autoencoder, encoder = build_autoencoder(IMG_HEIGHT, IMG_WIDTH, LATENT_DIM)
    autoencoder.summary()

    # 3. Trening
    print("\nRozpoczynam trening...")
    autoencoder.fit(train_ds, epochs=EPOCHS)
    print("Trening zakończony.\n")

    # 4. Wizualizacje
    visualize_reconstruction(autoencoder, train_ds)
    visualize_latent_space(encoder, train_ds)


if __name__ == "__main__":
    main()