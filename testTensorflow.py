import tensorflow as tf
import os
hello = tf.constant('hello, tensorflow')
sess = tf.Session()
print(sess.run(hello))
## Create a 2d grid, with some percentage of blockage, some percentage of node pairs, and among these node pairs,
#some are already connected. The purpose for it is to mimic the the fractional replay of a game.
##
