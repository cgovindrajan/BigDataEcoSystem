'''
referance used for adjusting modelling :
    tensorflow CNN example
    kaggle discussion forum
    Github: tgjeon:readme.md
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import sys as sys
import random

# Parameters Declaration Section
InterationCount = 3000
SPLIT_SIZE = 2800
BATCH_SIZE = 100
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.50
IMAGE_FACTORIZATION_CONSTANT = 1.0 /255.0

def generate_batch(batchSize,randomizeData):
    global train_images
    global train_labels
    global iterator

    begin = iterator
    iterator += batchSize
    if randomizeData == 1:
        if begin > limit:
            begin = random.randrange(1,limit)
            end = begin + BATCH_SIZE
            return train_images[begin:end], train_labels[begin:end]
    else:
        if iterator > limit:
            shuffVal = np.arange(limit)
            np.random.shuffle(shuffVal)
            train_images = train_images[shuffVal]
            train_labels = train_labels[shuffVal]
            begin = 0
            iterator = batchSize
    end = iterator
    return train_images[begin:end], train_labels[begin:end]

def parseLabels(labels):
	lables1D = np.zeros((32000,10))
	lables1D.flat[(np.arange(32000)*(10)) + labels.ravel()] = 1
	lables1D = lables1D.astype('int32')
	return lables1D

# Read MNIST data set (Train data from CSV file)
input_data = pd.read_csv(sys.argv[1])

images = np.multiply(((input_data.iloc[:,1:].values).astype('float')),IMAGE_FACTORIZATION_CONSTANT)

labels = parseLabels(input_data[[0]].values.ravel())

validation_images = images[:SPLIT_SIZE]
validation_labels = labels[:SPLIT_SIZE]

train_images = images[SPLIT_SIZE:]
train_labels = labels[SPLIT_SIZE:]
iterator = 0
limit = train_images.shape[0]
randomizeData = sys.argv[3]

Xval = tf.placeholder('float', shape=[None, 784])
Yval = tf.placeholder('float', shape=[None, 10])
dropconvVal = tf.placeholder('float')
drophiddenVal = tf.placeholder('float')

layer1Weight = tf.get_variable("layer1Weight", shape=[5, 5, 1, 32], initializer=tf.random_uniform_initializer(-tf.sqrt(6.0 / (25 + 32)), tf.sqrt(6.0 / (25 + 32))))
layer2Weight = tf.get_variable("layer2Weight", shape=[5, 5, 32, 64], initializer=tf.random_uniform_initializer(-tf.sqrt(6.0 / (800 + 64)), tf.sqrt(6.0 / (800 + 64))))
layer3Weight = tf.get_variable("layer3Weight", shape=[64*7*7, 4096], initializer=tf.random_uniform_initializer(-tf.sqrt(6.0 / (3136 + 4096)), tf.sqrt(6.0 / (3136 + 4096))))
layer4Weight = tf.get_variable("layer4Weight", shape=[4096, 10],initializer=tf.random_uniform_initializer(-tf.sqrt(6.0 / (4096 + 10)), tf.sqrt(6.0 / (4096 + 10))))

layer1Biases = tf.Variable(tf.constant(0.1, shape=[32]))
layer2Biases = tf.Variable(tf.constant(0.1, shape=[64]))
layer3Biases = tf.Variable(tf.constant(0.1, shape=[4096]))
layer4Biases = tf.Variable(tf.constant(0.1, shape=[10]))

XvalReshape = tf.reshape(Xval, [-1,28 , 28,1])

layer1con = tf.nn.relu(tf.nn.conv2d(XvalReshape, layer1Weight, strides=[1, 1, 1, 1], padding='SAME') + layer1Biases)
layer1pool = tf.nn.max_pool(layer1con, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer1drop = tf.nn.dropout(layer1pool, dropconvVal)

layer2con = tf.nn.relu(tf.nn.conv2d(layer1drop, layer2Weight, strides=[1, 1, 1, 1], padding='SAME')+ layer2Biases)
layer2pool = tf.nn.max_pool(layer2con, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

layer3flat = tf.reshape(layer2pool, [-1, layer3Weight.get_shape().as_list()[0]])
layer3feed = tf.nn.relu(tf.matmul(layer3flat, layer3Weight)+ layer3Biases)
layer3drop = tf.nn.dropout(layer3feed, drophiddenVal)

YPredicate = tf.nn.softmax(tf.matmul(layer3drop, layer4Weight)+ layer4Biases)

entropVal = -tf.reduce_sum(Yval*tf.log(YPredicate))
networkTrain = tf.train.AdamOptimizer(1e-4).minimize(entropVal)
networkAccuracy = tf.reduce_mean(tf.cast((tf.equal(tf.argmax(YPredicate, 1), tf.argmax(Yval, 1))), 'float'))
networkpredict = tf.argmax(YPredicate, 1)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(InterationCount):
    x, y = generate_batch(BATCH_SIZE,randomizeData)
    if i%100 == 0:
        print('Processing Iteration No :: %d'%i)
    sess.run(networkTrain, feed_dict={Xval: x, Yval: y, dropconvVal: DROPOUT_CONV, drophiddenVal: DROPOUT_HIDDEN})

validation_networkAccuracy = networkAccuracy.eval(feed_dict={Xval: validation_images,Yval: validation_labels,
                                                   dropconvVal: DROPOUT_CONV, drophiddenVal: DROPOUT_HIDDEN})
print('Validation networkAccuracy :: %.6f'%validation_networkAccuracy)

test_images = (pd.read_csv(sys.argv[2]).values).astype('float')
test_images = np.multiply(test_images, IMAGE_FACTORIZATION_CONSTANT)
size = test_images.shape[0]

networkpredicted_lables = np.zeros(test_images.shape[0])
for i in range(0,size//BATCH_SIZE):
    j = i + 1
    networkpredicted_lables[i*BATCH_SIZE : (j)*BATCH_SIZE] = networkpredict.eval(feed_dict={Xval: test_images[i*BATCH_SIZE : (j)*BATCH_SIZE], dropconvVal: 1.0, drophiddenVal: 1.0})

np.savetxt('result.csv',
               np.c_[range(1, len(test_images) + 1), networkpredicted_lables],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')
sess.close()
