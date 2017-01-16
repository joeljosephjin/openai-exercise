import tensorflow as tf

q_lr = 0.001
target_lr = 0.1

with tf.name_scope('input'):
    observation = tf.placeholder(tf.float32, [None,6], 'observation')
    target = tf.placeholder(tf.float32, [None,1], 'target')

with tf.name_scope('q_net'):
    w_q1 = tf.Variable(tf.truncated_normal([6,18], stddev=0.1),
    name='w_q1')
    b_q1 = tf.Variable(tf.constant(0.1, shape=[18]), name='b_q1')
    pa_q1 = tf.matmul(observation, w_q1) + b_q1
    a_q1 = tf.nn.relu(pa_q1, name='a_q1')

    w_q2 = tf.Variable(tf.truncated_normal([18,1], stddev=1),
    name='w_v2')
    b_q2 = tf.Variable(tf.constant(0.1), name='b_q2')
    value = tf.matmul(a_q1, w_q2) + b_q2
    mse = tf.contrib.losses.mean_squared_error(value, target)
    train_value = tf.train.AdamOptimizer(learning_rate=q_lr)\
    .minimize(mse)
