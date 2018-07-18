
# coding: utf-8

# In[9]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
def weight_init(shape):
    weight=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weight)
def bias_init(shape):
    bias=tf.constant(0.2,shape=shape)
    return tf.Variable(bias)
def conv_d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
w_conv1=weight_init([5,5,1,32])
b_conv1=bias_init([32])
x_image=tf.reshape(x,[-1,28,28,1])
f_conv2d=tf.nn.relu(conv_d(x_image,w_conv1)+b_conv1)
f_pool=max_pool(f_conv2d)
w_conv2=weight_init([5,5,32,64])
b_conv2=bias_init([64])
s_conv2d=tf.nn.relu(conv_d(f_pool,w_conv2) + b_conv2)
s_pool=max_pool(s_conv2d)
weight_flat=weight_init([7*7*64,1024])
bias_flat=bias_init([1024])
s_pool_flat=tf.reshape(s_pool,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(s_pool_flat,weight_flat) + bias_flat)
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
weight_flat2=weight_init([1024,10])
bias_flat2=bias_init([10])
y_conv=tf.matmul(h_fc1_drop,weight_flat2) + bias_flat2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.InteractiveSession()
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

