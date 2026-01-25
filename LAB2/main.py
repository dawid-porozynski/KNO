import argparse
import tensorflow as tf
import numpy as np


def rotate_points(points, degrees):
    # Zamiana stopni na radiany
    theta = degrees * np.pi / 180.0

    # Macierz obrotu 2x2
    # [[cos, -sin],
    #  [sin,  cos]]
    # Skleja wartości w jeden tensor
    rotation_matrix = tf.stack(
        [[tf.cos(theta), -tf.sin(theta)], [tf.sin(theta), tf.cos(theta)]]
    )

    # Bezpiecznik typów danych
    points = tf.cast(points, dtype=tf.float32)

    # Mnożenie macierzy
    return tf.matmul(points, rotation_matrix)

'''
zad3
def solve_linear(A, b):
    A = tf.cast(A, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)
    return tf.linalg.solve(A, b)

'''
def main():
    # zad5
    parser = argparse.ArgumentParser(description="Lab 2: Tensory")
    parser.add_argument("--mode", choices=["rotate", "solve"], required=True)
    parser.add_argument(
        "--data",
        nargs="+",
        type=float,
        required=True,
        help="Dane liczbowe (ciąg liczb)",
    )

    # Argumenty dodatkowe
    parser.add_argument(
        "--angle", type=float, default=90, help="Kąt obrotu (dla rotate)"
    )
    parser.add_argument("--size", type=int, help="Rozmiar macierzy N (dla solve)")

    args = parser.parse_args()

    # logika
    if args.mode == "rotate":
        # Zamiana płaskiej listy na pary (x, y)
        points = np.array(args.data).reshape(-1, 2)
        # Zamiana zwykłej tablicy na tenskor
        points_tf = tf.constant(points)

        result = rotate_points(points_tf, args.angle)
        print(f"\nWynik obrotu o {args.angle}°:\n", result.numpy())


    elif args.mode == "solve":

        if len(args.data) != 3:
            print("Musza byc 3 wejscia")

            return

        a = tf.constant(args.data[0], dtype=tf.float32)
        b = tf.constant(args.data[1], dtype=tf.float32)
        c = tf.constant(args.data[2], dtype=tf.float32)

        delta = tf.square(b) - 4.0 * a * c

        print(f"Równanie: {a.numpy()}x^2 + {b.numpy()}x + {c.numpy()} = 0")

        print(f"Delta: {delta.numpy()}")

        if delta < 0:
            print("Brak rozwiązań")

        elif delta == 0:

            x0 = -b / (2.0 * a)
            print(f"Jedno rozwiązanie: x0 = {x0.numpy()}")

        else:
            sqrt_delta = tf.sqrt(delta)
            x1 = (-b - sqrt_delta) / (2.0 * a)
            x2 = (-b + sqrt_delta) / (2.0 * a)
            print(f"Dwa rozwiązania:\nx1 = {x1.numpy()}\nx2 = {x2.numpy()}")

if __name__ == "__main__":
    main()