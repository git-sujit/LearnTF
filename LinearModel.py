"""
Example of Linear Model
"""
import tensorflow as tf

weight = tf.Variable([0.3], dtype=tf.float32)
bias = tf.Variable([-0.3], dtype=tf.float32)
ip = tf.placeholder(tf.float32)

linear_model = weight * ip + bias
init = tf.global_variables_initializer()

sess = tf.Session();
sess.run(init)

print(sess.run(linear_model, {ip: [1, 2, 3, 4]}))

desired_output = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - desired_output)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {ip: [1, 2, 3, 4], desired_output: [0, -1, -2, -3]}))

# For weight=-1 and bias=1, this will converge.
