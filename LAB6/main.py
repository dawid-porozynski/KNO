import tensorflow as tf
import matplotlib.pyplot as plt

DATA_DIR = "dataset_images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
LATENT_DIM = 128
EPOCHS = 20


def main():
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        label_mode=None,  # Usuwamy etykiety
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Modyfikacja obrazu
    aug_layers = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),  # normalizacja
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )
    # Warstwa tylko do normalizacji
    norm_layer = tf.keras.layers.Rescaling(1.0 / 255)

    # niekształcony obraz i czysty obraz
    ds = ds.map(lambda x: (aug_layers(x), norm_layer(x)))

    # Osobny Enkoder
    encoder_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    encoder_output = tf.keras.layers.Dense(LATENT_DIM, name="latent_space")(x)  # Wąskie gardło

    # model enkodera
    encoder = tf.keras.models.Model(encoder_input, encoder_output, name="encoder")

    # Osobny dekoder
    decoder_input = tf.keras.Input(shape=(LATENT_DIM,))
    # Dekoder rekonstrukcja
    x = tf.keras.layers.Dense(32 * 32 * 64, activation="relu")(decoder_input)
    x = tf.keras.layers.Reshape((32, 32, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_output = tf.keras.layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

    # model dekodera
    decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder")

    # Polaczenie do treningu
    training_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    latent_code = encoder(training_input)
    reconstruction = decoder(latent_code)

    # Model pomocniczy
    trainer = tf.keras.models.Model(training_input, reconstruction, name="trainer")

    trainer.compile(optimizer="adam", loss="mean_squared_error")
    trainer.summary()

    trainer.fit(ds, epochs=EPOCHS)

    sample_batch = next(iter(ds))
    original_imgs = sample_batch[0][:5]

    # Uzycie osobnych modeli
    codes = encoder.predict(original_imgs)  # Enkoder
    reconstructed_imgs = decoder.predict(codes)  # Dekoder

    print({codes.shape})
    print(codes[0])

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # gorny rzad
        plt.subplot(2, 5, i + 1)
        plt.imshow(original_imgs[i])
        plt.title("Wejście")
        plt.axis("off")

        # dolny rzad
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(reconstructed_imgs[i])
        plt.title("Wynik")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()