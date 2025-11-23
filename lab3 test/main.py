# wine_classifier_final.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

#wczytanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
col_names = ["Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
             "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
             "Color_intensity", "Hue", "OD280_OD315", "Proline"]

df = pd.read_csv(url, header=None, names=col_names)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
y_int = df.iloc[:, 0].to_numpy(dtype=np.int64) - 1
y_one_hot = tf.keras.utils.to_categorical(y_int, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
)


def build_model1():
    model = tf.keras.Sequential([
        layers.Input(shape=(13,), name="input"),
        layers.Dense(128, name="fc1"),
        layers.BatchNormalization(name="bn1"),
        layers.ReLU(name="relu1"),
        layers.Dropout(0.4, name="drop1"),
        layers.Dense(64, name="fc2"),
        layers.BatchNormalization(name="bn2"),
        layers.ReLU(name="relu2"),
        layers.Dropout(0.3, name="drop2"),
        layers.Dense(32, name="fc3"),
        layers.ReLU(name="relu3"),
        layers.Dense(3, activation="softmax", name="output")
    ], name="Wide_ReLU_BN_Dropout")
    return model


def build_model2():
    model = tf.keras.Sequential([
        layers.Input(shape=(13,), name="input"),
        layers.Dense(64, activation="gelu", kernel_initializer="glorot_uniform", name="gelu1"),
        layers.Dense(48, activation="gelu", kernel_initializer="glorot_uniform", name="gelu2"),
        layers.Dense(32, activation="gelu", kernel_initializer="glorot_uniform", name="gelu3"),
        layers.Dense(24, activation="gelu", kernel_initializer="glorot_uniform", name="gelu4"),
        layers.Dense(16, activation="gelu", kernel_initializer="glorot_uniform", name="gelu5"),
        layers.Dense(3, activation="softmax", name="output")
    ], name="Deep_GELU")
    return model



import datetime

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15),
    tensorboard_cb
]

print("Trening modelu 1")
model1 = build_model1()
model1.compile(optimizer=tf.keras.optimizers.Adam(0.003),
               loss="categorical_crossentropy", metrics=["accuracy"])
history1 = model1.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=400, batch_size=32, callbacks=callbacks, verbose=2)

print("\nTrening modelu 2")
model2 = build_model2()
model2.compile(optimizer=tf.keras.optimizers.Adam(0.004),
               loss="categorical_crossentropy", metrics=["accuracy"])
history2 = model2.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=400, batch_size=16, callbacks=callbacks, verbose=2)

#wizualizacja
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history1.history['val_accuracy'], label=f"Model 1 (max {max(history1.history['val_accuracy']):.4f})")
plt.plot(history2.history['val_accuracy'], label=f"Model 2 (max {max(history2.history['val_accuracy']):.4f})")
plt.title("Validation Accuracy")
plt.xlabel("Epoka");
plt.ylabel("Accuracy")
plt.legend();
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label="Model 1 train loss")
plt.plot(history2.history['loss'], label="Model 2 train loss")
plt.title("Training Loss")
plt.xlabel("Epoka");
plt.ylabel("Loss")
plt.legend();
plt.grid(True)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=200)
plt.show()

best_val_acc1 = max(history1.history['val_accuracy'])
best_val_acc2 = max(history2.history['val_accuracy'])

if best_val_acc1 >= best_val_acc2:
    best_model = model1
    print(f"\nbest model: Model 1 (val_acc = {best_val_acc1:.4f})")
else:
    best_model = model2
    print(f"\nbest model: Model 2 (val_acc = {best_val_acc2:.4f})")

best_model.save("best_wine_model.keras")


#parser
def predict_wine():
    parser = argparse.ArgumentParser(description="Predykcja klasy wina")
    parser.add_argument("--alcohol", type=float, required=True)
    parser.add_argument("--malic_acid", type=float, required=True)
    parser.add_argument("--ash", type=float, required=True)
    parser.add_argument("--alcalinity", type=float, required=True)
    parser.add_argument("--magnesium", type=float, required=True)
    parser.add_argument("--total_phenols", type=float, required=True)
    parser.add_argument("--flavanoids", type=float, required=True)
    parser.add_argument("--nonflavanoid_phenols", type=float, required=True)
    parser.add_argument("--proanthocyanins", type=float, required=True)
    parser.add_argument("--color_intensity", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--od280_od315", type=float, required=True)
    parser.add_argument("--proline", type=float, required=True)

    args = parser.parse_args()

    model = tf.keras.models.load_model("best_wine_model.keras")
    sample = np.array([[args.alcohol, args.malic_acid, args.ash, args.alcalinity,
                        args.magnesium, args.total_phenols, args.flavanoids,
                        args.nonflavanoid_phenols, args.proanthocyanins,
                        args.color_intensity, args.hue, args.od280_od315, args.proline]],
                      dtype=np.float32)

    pred = model.predict(sample, verbose=0)
    predicted_class = np.argmax(pred, axis=1)[0] + 1  # zwracamy 1,2,3
    confidence = np.max(pred)

    print(f"\nPredicted class: {predicted_class}")
    print(f"confidence: {confidence:.1%}")


if __name__ == "__main__":

    pass
