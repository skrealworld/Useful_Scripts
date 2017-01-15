#!/usr/bin/python2

from __future__ import print_function


import pickle
import numpy as np
import tensorflow as tf


def get_batch(data, labels, batch_size, itr):
    """

    """
    start_idx = (itr * batch_size)
    end_idx = (itr * batch_size + batch_size)

    if end_idx<len(data):
        return data[start_idx:end_idx],labels[start_idx:end_idx] 
    else: 
	return data[0:batch_size],labels[0:batch_size] 


if __name__ == "__main__":

    batch_size = 5
    DENSE_WEIGHTS = 385
    iterations = 10
    lr = 0.001
    training_samples = 350
    display_step = 1
    no_of_inputs = 173*13*5
    no_of_classes = 4
    dropout = 0.75  

    #Load saved picklefiles 
    data = pickle.load(open('dataset2','rb'))
    data = np.asarray(data)
    data = data.reshape((data.shape[0],no_of_inputs
        ))
    labels = pickle.load(open('labels2','rb'))

    #Shuffle the data
    permutation = np.random.permutation(data.shape[0])
    #data = data[permutation]
    old_labels = labels
    old_data = data
    for x in range(0,data.shape[0]):
        labels[x] = old_labels[permutation[x]]
        data[x] = old_data[permutation[x]] 

    # Split Train/Test
    trainData = data[:training_samples]
    trainLabels = labels[:training_samples]

    testData = data[training_samples:]
    testLabels = labels[training_samples:]


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, no_of_inputs])
    y = tf.placeholder(tf.float32, [None, no_of_classes])
    keep_prob = tf.placeholder(tf.float32) 


    # Create model
    def conv2d(sound, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))

    def max_pool(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, 173, 13, 5])

        # Convolution Layer
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = max_pool(conv1, k=4)
        # Apply Dropout
        conv1 = tf.nn.dropout(conv1, _dropout)

        # Convolution Layer
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = max_pool(conv2, k=2)
        # Apply Dropout
        conv2 = tf.nn.dropout(conv2, _dropout)

        # Convolution Layer
        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = max_pool(conv3, k=2)
        # Apply Dropout
        conv3 = tf.nn.dropout(conv3, _dropout)

        # Fully connected layer

        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        #dense1 = tf.reshape(conv3, [-1,pool5Shape[1]*pool5Shape[2]*pool5Shape[3]])
        # Relu activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        # Apply Dropout
        dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out


    # Store layers weight & bias
    weights = {
        # 4x4 conv, 1 input, 149 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        # 4x4 conv, 149 inputs, 73 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        # 4x4 conv, 73 inputs, 35 outputs
        'wc3': tf.Variable(tf.random_normal([2, 2, 73, 35])),
        # fully connected, 38*8*35 inputs, 2^13 outputs
        'wd1': tf.Variable(tf.random_normal([DENSE_WEIGHTS, 512])),
        # 2^13 inputs, 13 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([512, no_of_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([149])+0.01),
        'bc2': tf.Variable(tf.random_normal([73])+0.01),
        'bc3': tf.Variable(tf.random_normal([35])+0.01),
        'bd1': tf.Variable(tf.random_normal([512])+0.01),
        'out': tf.Variable(tf.random_normal([no_of_classes])+0.01)
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("Training acc : ", accuracy ) 
    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < iterations:
            batch_xs, batch_ys = get_batch(trainData, trainLabels, batch_size, step)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
            step += 1

        save_path = saver.save(sess, "trained_model.final")
        print("Saving model : %s" % save_path)

        # Calculate accuracy
        print("Testing acc:", sess.run(accuracy, feed_dict={x: testData,
                                                                 y: testLabels,
                                                                 keep_prob: 1.}))
