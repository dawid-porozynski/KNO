import tensorflow as tf
import numpy as np


def rotate_points(points, degrees):

    theta = tf.constant(degrees * np.pi / 180, dtype=tf.float32)

    # macierz 2x2
    rotation_matrix = tf.stack([
        [tf.cos(theta), -tf.sin(theta)],
        [tf.sin(theta),  tf.cos(theta)]
    ])

    # mnożenie macierzy:
    rotated_points = tf.matmul(points, rotation_matrix)

    return rotated_points

def solve_linear(A, b):
    A = tf.cast(A, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)

    x = tf.linalg.solve(A, b)

    return x

def main():

    points1 = tf.constant([
        [3.0, 0.0],
        [1.0, 2.0],
        [4.0, 5.0]
    ])
    points2 = tf.constant([
        [1.0, 3.0],
        [2.0, 1.0],
        [3.0, 1.0]
    ])

    rotated1 = rotate_points(points1, 90)
   # rotated2 = rotate_points(points2, 90)

    A = tf.constant([[2, 1],
                     [1, 3]], dtype=tf.float32)

    b = tf.constant([[5],
                     [10]], dtype=tf.float32)

    solution = solve_linear(A, b)

    print("\nObrócone punkty o 90°:")
    print(tf.round(rotated1))
   # print(tf.round(rotated2))

    print("\nRozwiazanie liniowe:")
    print(tf.round(solution))

if __name__ == "__main__":
    main()
