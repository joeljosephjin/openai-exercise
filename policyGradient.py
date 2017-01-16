import tensorflow as tf

value_lr = 0.1
target_lr = 0.1

with tf.name_scope('input'):
    observation = tf.placeholder(tf.float32, [None,2], 'observation')
    target = tf.placeholder(tf.float32, [None,1], 'target')
    with tf.name_scope('action_net'):
        action = tf.placeholder(tf.float32, [None,3], 'action')

with tf.name_scope('value_net'):
    w_v1 = tf.Variable(tf.truncated_normal([2,6], stddev=0.1),
    name='w_v1')
    b_v1 = tf.Variable(tf.constant(0.1, shape=[6]), name='b_v1')
    pa_v1 = tf.matmul(observation, w_v1) + b_v1
    a_v1 = tf.nn.relu(pa_v1, name='a_v1')

    w_v2 = tf.Variable(tf.truncated_normal([6,1], stddev=0.1),
    name='w_v2')
    b_v2 = tf.Variable(tf.constant(0.1), name='b_v2')
    value = tf.matmul(a_v1, w_v2) + b_v2
    mse = tf.contrib.losses.mean_squared_error(value, target)
    train_value = tf.train.AdamOptimizer(learning_rate=value_lr)\
    .minimize(mse)

with tf.name_scope('action_net'):
    w_a1 = tf.Variable(tf.truncated_normal([2,6], stddev=0.1),
    name='w_a1')
    b_a1 = tf.Variable(tf.constant(0.1, shape=[6]), name='b_a1')
    pa_a1 = tf.matmul(observation, w_a1) + b_a1
    a_a1 = tf.nn.relu(pa_a1, name='a_a1')

    w_a2 = tf.Variable(tf.truncated_normal([6,3], stddev=1),
    name='w_a2')
    b_a2 = tf.Variable(tf.constant(0.1, shape=[3]), name='b_a2')
    pa_a2 = tf.matmul(a_a1, w_a2) + b_a2
    action_prob = tf.nn.softmax(pa_a2, name='action_prob')

    with tf.name_scope('gradient'):
        chosen_action_prob = tf.reduce_sum(tf.multiply(action_prob, action),
        axis=1, keep_dims=True)
        log_prob = tf.log(chosen_action_prob)
        J = tf.multiply(log_prob, target-value)
        loss = - tf.reduce_mean(J)
        train_action = tf.train.AdamOptimizer(learning_rate=target_lr)\
        .minimize(loss)
