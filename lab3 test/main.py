import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

DATA_DIR = 'dataset_images'
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
LATENT_DIM = 2
EPOCHS = 50


def main():
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        label_mode=None,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    aug_layers = tf.keras.Sequential([
        layers.Rescaling(1. / 255),  #normalizacja
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    # Warstwa tylko do normalizacji
    norm_layer = layers.Rescaling(1. / 255)

    # Dzięki temu sieć uczy się "naprawiać" i generować ładne obrazki
    ds = ds.map(lambda x: (aug_layers(x), norm_layer(x)))

    # Budowa Autoenkodera
    input_img = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Enkoder
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(input_img)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(LATENT_DIM, name="latent_space")(x)  # Wąskie gardło (2 wymiary)

    x = layers.Dense(32 * 32 * 64, activation='relu')(encoded)
    x = layers.Reshape((32, 32, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoded = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

    # Składamy model
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # Trening
    print("Rozpoczynam trening...")
    autoencoder.fit(ds, epochs=EPOCHS)

    sample_batch = next(iter(ds))

    augmented_imgs = sample_batch[0][:5]

    reconstructed = autoencoder.predict(augmented_imgs)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # gorny rzad
        plt.subplot(2, 5, i + 1)
        plt.imshow(augmented_imgs[i])
        plt.title("Wejście")
        plt.axis("off")

        # dolny rzad
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(reconstructed[i])
        plt.title("Wynik")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()