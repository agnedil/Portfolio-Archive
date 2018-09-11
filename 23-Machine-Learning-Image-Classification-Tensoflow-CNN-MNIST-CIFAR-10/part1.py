# AML UIUC
# used parts of the Tensorboard tutorial from https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
# used the Tensorboard quick start tutorial at https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
# used part of the above google tutorial in this repository https://github.com/martin-gorner/tensorflow-mnist-tutorial

import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


print("Loading data")
tf.set_random_seed(0)

# 60K images+labels to mnist.train and 10K images+labels to mnist.test
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# architecture
# input X: 28x28 grayscale images, 1st dim (None) = img index in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])                         # for correct answers
step = tf.placeholder(tf.int32)                                     # variable learning rate

print("Data  loaded")

# channel count
K = 16                                                               # 1st convolutional layer output depth
L = 32                                                              # 2nd conv. layer
M = 64                                                              # 3rd conv. layer
N = 200                                                             # fully connected layer (has 10 softmax neurons)

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))     # input X [batch, 28, 28, 1], stride 1
B1 = tf.Variable(tf.ones([K])/10)                                   # 5x5 patch, 1 input channel, K output channels
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))     # etc., stride 2
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))     # stride 2
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))   # fully connected layer (relu), W4 [7*7*12, 200], B4 [200]
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))          # fully connected layer (softmax), W5 [200, 10], B5 [10]
B5 = tf.Variable(tf.ones([10])/10)

# model
stride = 1                                                          # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2                                                          # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2                                                          # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from 3rd conv. layer for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ) normalized for batches of 100 img
# softmax_cross_entropy_with_logits - to avoid numerical stability problems with log(0) / NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
tf.summary.scalar('cross_entropy', cross_entropy)

# accuracy = [0, 1]
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# learning rate = 0.0001 + 0.003 * (1/e)^(step/2000)) = exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

# for Tensoboard statistics
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/andrew/tmp/board/train', sess.graph)
test_writer = tf.summary.FileWriter('/home/andrew/tmp/board/test', sess.graph)
#tf.global_variables_initializer().run()

# main tran - test cycle
for i in range(10000+1):

    batch_X, batch_Y = mnist.train.next_batch(100)                  # batches of 100 images

    if i % 20 == 0:                                                 # record train summaries and train
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, a, c, l = sess.run([merged, accuracy, cross_entropy, lr],
                                  feed_dict={X: batch_X, Y_: batch_Y, step: i},
                                    options = run_options,
                                    run_metadata = run_metadata)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)

    if i % 100 == 0:                                                # record summaries + test accuracy
        summary, a, c = sess.run([merged, accuracy, cross_entropy],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        test_writer.add_summary(summary, i)
    sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})                                # backpropagation training step

train_writer.close()
test_writer.close()
