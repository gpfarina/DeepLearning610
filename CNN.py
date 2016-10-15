import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split


# Load the data
X1 = np.load('X_Train.npy')
Y1 = np.load('Y_Train.npy')
print(X1.shape)

# Split the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(X1,Y1,test_size=0.75,random_state=1)


# Architecture of the CNN
# Input-image -> convolution_1 -> pooling_1 -> convolution_2 -> pooling_2 -> fully connected for classification

# Network parameters
classes = 2
neurons = 512
features = 32
dropout = 0.75
kernel = 5


# Create the place holders for input and output
X = tf.placeholder('float', [None, np.shape(x_train)[1]])
Y = tf.placeholder('float', [None, np.shape(y_train)[1]])

# Functions
def conv(x, W, b):
    strides = 1
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
    # here we adjust the filter which are weights. Isn't it brilliant!!
    # strides are nothing but the step taken by convolution filter
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool(x):
    # Max pooling uses the same concept as of convolution but it selects the max instead of convolution
    k = 2
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Model
def Model(x,W,b,dropout):
    # Reshape the vector into a matrix (original image)
    x = tf.reshape(x, shape=[-1, 40, 40, 1])
    # Convolution Layer 1
    conv1 = conv(x, W['cv1'], b['bv1'])
    # Max Pooling (down-sampling)
    mpool1 = maxpool(conv1)

    # Convolution Layer 2
    conv2 = conv(mpool1, W['cv2'], b['bv2'])
    # Max Pooling (down-sampling)
    mpool2 = maxpool(conv2)

    # Now the fully connected layer used for classification
    # flatten the mpool2 into a vector (remember it is a tensor now)
    input = tf.reshape(mpool2, [-1, W['fc1'].get_shape().as_list()[0]])
    h1 = tf.add(tf.matmul(input,W['fc1']),b['bc1'])
    h1 = tf.nn.relu(h1) # can change to sigmoid
    # Before passing to output layer apply drop out
    # Apply Dropout
    h1 = tf.nn.dropout(h1, dropout)

    # output
    output = tf.matmul(h1,W['fc2'])
    print(output)
    return output

# Create the weights (smart way of storing them in a dictionary, I like it!)
W = {
    # I am using a kernel size of 3,3 and output of
    'cv1': tf.Variable(tf.random_normal([kernel, kernel, 1, features])),
    'cv2': tf.Variable(tf.random_normal([kernel, kernel, features, features*2])),
    # Fully connected layer weights
    'fc1': tf.Variable(tf.random_normal([10*10*features*2, neurons])),
    'fc2': tf.Variable(tf.random_normal([neurons, np.shape(y_train)[1]]))
}

b = {
    'bv1': tf.Variable(tf.random_normal([features])),
    'bv2': tf.Variable(tf.random_normal([features*2])),
    'bc1': tf.Variable(tf.random_normal([neurons])),
    # 'out': tf.Variable(tf.random_normal([classes]))
}

# That was long!, now set up the model and run it
# Optimisation step
M = Model(X,W,b,dropout)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(M,Y))
train_op = tf.train.AdagradOptimizer(0.01).minimize(cost)
Predicted = tf.nn.sigmoid(Model(X,W,b,dropout))

batchsize = 128 # We need not have a batch size as the data size is very less

# Run the optimisation
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(50000):
        p = np.random.permutation(range(len(x_train)))
        trX, trY = x_train[p], y_train[p]
        for j in range(0, len(x_train), batchsize):
            last = j + batchsize
            sess.run(train_op, feed_dict={X: trX[j:j + batchsize], Y: trY[j:j + batchsize]})
        if i%1000 == 0:
            # can you figure out how to run our model for different input?

        
            # print(Model(x_test,sess.run(W),sess.run(b),0.4)) # This line is giving error
            # # Predicted = np.round(Predicted)
            # # print('Testing', np.mean(Predicted == y_test), sum(y_test) / len(y_test))
