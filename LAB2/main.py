# lab2_final.py
import argparse
import tensorflow as tf
import numpy as np


# --- ZADANIE 2: Funkcja obrotu (zoptymalizowana przez tf.function) ---
@tf.function
def rotate_points(points, degrees):
    # Zamiana stopni na radiany
    theta = degrees * np.pi / 180.0

    # Macierz obrotu 2x2
    # [[cos, -sin],
    #  [sin,  cos]]
    rotation_matrix = tf.stack([
        [tf.cos(theta), -tf.sin(theta)],
        [tf.sin(theta), tf.cos(theta)]
    ])

    # Bezpiecznik typów danych
    points = tf.cast(points, dtype=tf.float32)

    # Mnożenie macierzy
    return tf.matmul(points, rotation_matrix)


# --- ZADANIE 3: Rozwiązywanie równań ---
@tf.function
def solve_linear(A, b):
    A = tf.cast(A, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)
    return tf.linalg.solve(A, b)


def main():
    # --- ZADANIE 5: Obsługa argumentów ---
    parser = argparse.ArgumentParser(description="Lab 2: Tensory")
    parser.add_argument("--mode", choices=["rotate", "solve"], required=True,
                        help="Tryb: rotate (obrót) lub solve (równania)")
    parser.add_argument("--data", nargs="+", type=float, required=True, help="Dane liczbowe (ciąg liczb)")

    # Argumenty dodatkowe
    parser.add_argument("--angle", type=float, default=90, help="Kąt obrotu (dla rotate)")
    parser.add_argument("--size", type=int, help="Rozmiar macierzy N (dla solve)")

    args = parser.parse_args()

    # --- Logika programu ---
    if args.mode == "rotate":
        # Zamiana płaskiej listy na pary (x, y)
        points = np.array(args.data).reshape(-1, 2)
        points_tf = tf.constant(points)

        result = rotate_points(points_tf, args.angle)
        print(f"\nWynik obrotu o {args.angle}°:\n", result.numpy())

    elif args.mode == "solve":
        if not args.size:
            print("Błąd: Podaj rozmiar macierzy parametrem --size N")
            return

        n = args.size
        # Pierwsze N*N liczb to Macierz A, reszta to Wektor b
        matrix_data = args.data[:n * n]
        vector_data = args.data[n * n:]

        A = tf.reshape(tf.constant(matrix_data), (n, n))
        b = tf.reshape(tf.constant(vector_data), (n, 1))

        try:
            result = solve_linear(A, b)
            print(f"\nRozwiązanie układu {n}x{n}:\n", result.numpy())
        except Exception:
            print("Błąd: Układ sprzeczny lub nieoznaczony (det=0).")


if __name__ == "__main__":
    main()