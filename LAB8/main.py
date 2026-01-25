import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Konfiguracja
LOOK_BACK = 20
EPOCHS = 10


def create_dataset(dataset):
    """Pomocnicza funkcja do tworzenia okien czasowych."""
    dataX, dataY = [], []
    for i in range(len(dataset) - LOOK_BACK):
        a = dataset[i:(i + LOOK_BACK)]
        dataX.append(a)
        dataY.append(dataset[i + LOOK_BACK])
    return np.array(dataX), np.array(dataY)


def main():
    # Generowanie danych sinus
    # Tworzymy 1000 punktów od 0 do 50
    x_axis = np.linspace(0, 50, 1000)
    dataset = np.sin(x_axis)

    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # Przygotowanie danych pod LSTM
    trainX, trainY = create_dataset(train_data)
    testX, testY = create_dataset(test_data)

    # LSTM wymaga: próbki, kroki, cechy
    trainX = np.reshape(trainX, (trainX.shape[0], LOOK_BACK, 1))
    testX = np.reshape(testX, (testX.shape[0], LOOK_BACK, 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(LOOK_BACK, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=32, verbose=1)

    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, dataset, label='Oryginał', color='lightgray', linestyle='--')

    train_x_range = x_axis[LOOK_BACK: len(train_predict) + LOOK_BACK]
    plt.plot(train_x_range, train_predict, label='Trening', color='blue')

    test_start = len(train_predict) + (LOOK_BACK * 2)
    test_x_range = x_axis[test_start: test_start + len(test_predict)]

    plt.plot(test_x_range, test_predict, label='Predykcja (Test)', color='red')

    plt.title("Predykcja funkcji Sinus (LSTM)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()