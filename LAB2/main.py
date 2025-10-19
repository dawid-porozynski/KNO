import tensorflow as tf
import numpy as np

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)