import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


model_path = "model.keras"


def train_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, epochs=5)  # użyj verbose=0 jeśli jest problem z konsolą
    model.evaluate(x_test, y_test)

    model.save(model_path)

    plt.plot(history.history["accuracy"], label="accuracy (train)")
    plt.plot(history.history["val_accuracy"], label="accuracy (val)")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.title("Krzywa uczenia")
    plt.legend()
    plt.show()
    return model


def load_model():
    return tf.keras.models.load_model(model_path)


def predict_image(model, image_path):
    image = tf.keras.utils.load_img(
        image_path, color_mode="grayscale", target_size=(28, 28)
    )
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions)
    print(f"Rozpoznana cyfra: {predicted_class}")
    return predicted_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    if os.path.exists(model_path):
        model = load_model()
    else:
        model = train_model()

    if args.image:
        predict_image(model, args.image)


if __name__ == "__main__":
    main()
