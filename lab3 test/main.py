import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.model_selection import train_test_split

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    col_names = ["Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
                 "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                 "Color_intensity", "Hue", "OD280_OD315", "Proline"]

    df = pd.read_csv(url, header=None, names=col_names)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_int = df.iloc[:, 0].to_numpy(dtype=np.int64) - 1
    y_one_hot = tf.keras.utils.to_categorical(y_int, num_classes=3)

    return train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot)


def build_model1():
    return tf.keras.Sequential([
        layers.Input(shape=(13,), name="input"),
        layers.BatchNormalization(name="bn_input"),  # Auto-skalowanie
        layers.Dense(128, name="fc1"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.4),
        layers.Dense(64, name="fc2"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax", name="output")
    ], name="Model_BN_Dropout")


def build_model2():
    return tf.keras.Sequential([
        layers.Input(shape=(13,), name="input"),
        layers.BatchNormalization(name="bn_input"),  # Auto-skalowanie
        layers.Dense(64, activation="gelu"),
        layers.Dense(48, activation="gelu"),
        layers.Dense(32, activation="gelu"),
        layers.Dense(16, activation="gelu"),
        layers.Dense(3, activation="softmax", name="output")
    ], name="Model_Deep_GELU")


def train_mode():
    print("Trening start:")
    X_train, X_test, y_train, y_test = load_data()

    # Callbacks (EarlyStopping zapobiega marnowaniu czasu)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]

    # Trening Modelu 1
    print(" > Trenuję Model 1...", end=" ", flush=True)
    model1 = build_model1()
    model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model1.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=200, batch_size=32, callbacks=callbacks, verbose=0)
    acc1 = max(h1.history['val_accuracy'])
    print(f"Gotowe (Acc: {acc1:.2%})")

    # Trening Modelu 2
    print(" > Trenuję Model 2...", end=" ", flush=True)
    model2 = build_model2()
    model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model2.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=200, batch_size=16, callbacks=callbacks, verbose=0)
    acc2 = max(h2.history['val_accuracy'])
    print(f"Gotowe (Acc: {acc2:.2%})")

    # Wykresy
    plt.figure(figsize=(10, 5))
    plt.plot(h1.history['val_accuracy'], label=f"Model 1 (max {acc1:.2f})")
    plt.plot(h2.history['val_accuracy'], label=f"Model 2 (max {acc2:.2f})")
    plt.title("Porównanie dokładności")
    plt.xlabel("Epoka")
    plt.legend()
    plt.grid(True)
    plt.savefig("results.png")
    print("\n[INFO] Wykres zapisano jako 'results.png'")

    # Zapis zwycięzcy
    if acc1 >= acc2:
        best_model = model1
        print(f"[INFO] Zwyciężył Model 1. Zapisuję...")
    else:
        best_model = model2
        print(f"[INFO] Zwyciężył Model 2. Zapisuję...")

    best_model.save("best_wine_model.keras")
    print("[SUKCES] Model gotowy do użycia.")


#predykcja
def predict_mode():
    if not os.path.exists("best_wine_model.keras"):
        print("Błąd: Nie znaleziono modelu! Uruchom program bez argumentów, aby wytrenować sieć.")
        return

    parser = argparse.ArgumentParser()
    # Argumenty wina
    args_list = ["alcohol", "malic_acid", "ash", "alcalinity", "magnesium",
                 "total_phenols", "flavanoids", "nonflavanoid_phenols",
                 "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"]
    for arg in args_list:
        parser.add_argument(f"--{arg}", type=float, required=True)

    args = parser.parse_args()

    # Ładowanie i predykcja
    model = tf.keras.models.load_model("best_wine_model.keras")
    sample = np.array([[getattr(args, a) for a in args_list]], dtype=np.float32)

    pred = model.predict(sample, verbose=0)
    klasa = np.argmax(pred) + 1
    pewnosc = np.max(pred)

    print(f"\n--- WYNIK ANALIZY ---")
    print(f"Klasa wina: {klasa}")
    print(f"Pewność:    {pewnosc:.1%}")


# --- MAIN ---
if __name__ == "__main__":
    # Jeśli są argumenty -> Predykcja. Jeśli nie ma -> Trening.
    if len(sys.argv) > 1:
        predict_mode()
    else:
        train_mode()