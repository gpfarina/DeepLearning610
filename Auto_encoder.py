import tensorflow as tf
import numpy as np

# Training data


# Testing data

# Set up the model
X = tf.placeholder('float', [None, np.shape(x_train)[1]])
Y = tf.placeholder('float', [None, np.shape(x_train)[1]])

#initial value of weights

layer_size = 200

w_h = tf.Variable(tf.random_uniform([np.shape(x_train)[1],layer_size],0,1)) # 200 neurons
w_o = tf.Variable(tf.random_uniform([layer_size,np.shape(x_train)[1]],0,1)) # Output layer

# Model
def model(X, w_h, w_o):
    h = tf.nn.tanh(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Optimisation step
x = model(X, w_h, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x, Y))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_op = tf.argmax(x, 1)

batchsize = 256

# Run the optimisation
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(20000):
        p = np.random.permutation(range(len(x_train)))
        trX, trY = x_train[p], y_train[p]
        for j in range(0,len(x_train),batchsize):
            last =  j + batchsize
            sess.run(train_op, feed_dict={X: x_train[j:j+batchsize], Y: x_train[j:j+batchsize]})

