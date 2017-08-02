"""
LINEAR REGRESSION USING GRADIENT DESCENT
TensorFlow provides optimizers that slowly changes each variable in order to minimize the loss function.
Simplest Optimizer: Gradient Descent
It modifies each variable according to the magnitude of the derivative of loss with respect to that variable
"""
# To avoid the warning : "The TensorFlow library wasn't compiled to use FMA instructions"
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Define weight/bias as variables and customer input as parameter
weight = tf.Variable([0.3], dtype=tf.float32)
bias = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
# Define Linear Model
linear_model = weight * x + bias
# Initialize variables
init = tf.global_variables_initializer()
# Get Tensor Session (Context to CPU/GPU)
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# Define desired output
desired_output = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - desired_output)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], desired_output: [0, -1, -2, -3]}))
# Optimizer: Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# Train the model
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], desired_output: [0, -1, -2, -3]})
# Print adjusted value of weight and bias
print(sess.run([weight, bias]))
print("Calculated Weight = ", sess.run(weight))
print("Calculated Bias = ", sess.run(weight))
