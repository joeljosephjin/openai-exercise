import tensorflow as tf

with name_scope('input'):
    observation = tf.placeholder(tf.float32, [None,4], 'observation')
    target = tf.placeholder(tf.float32, [None,1], 'target')
    with name_scope('action_net'):
        action = tf.placeholder(tf.float32, [None,2], 'action')

with name_scope('value_net'):
    weights_v1 = tf.Variable(tf.truncated_normal([4,10], stddev=0.1),
    name='weights_v1')
    biases_v1 = tf.Variable(0.1, shape=[10], name='biases_v1')
    preactivation_v1 = tf.matmul(observation, weights_v1) + biases_v1
    activation_v1 = tf.nn.relu(preactivation_v1, name='activation_v1')

    weights_v2 = tf.Variable(tf.truncated_normal([10,1], stddev=0.1),
    name='weights_v2')
    biases_v2 = tf.Variable(0.1, name='biases_v2')
    value = tf.matmul(activation_v1, weights_v2) + biases_v2
    mse = tf.contrib.losses.mean_squared_error(value, target)
    train_value = tf.train.AdamOptimizer(learning_rate=0.001)\
    .minimize(mse)

with name_scope('action_net'):
    weights_a1 = tf.Variable(tf.truncated_normal([4,1], stddev=0.1),
    name='weights_a1')
    #biases_a1 = tf.Variable(0.1 , name='biases_a1')
    preactivation_a1 = tf.matmul(observation, weights_a1) # + biases_a1
    # probability for action == 0
    action_prob = tf.sigmoid(preactivation_a1, name='action_prob')

    with name_scope('gradient'):
        sigmoid_derivative_0 = tf.sigmoid(-preactivation_a1)
        sigmoid_derivative_1 = -tf.sigmoid(preactivation_a1)
        sigmoid_derivative = tf.concat(1,
        [sigmoid_derivative_0, sigmoid_derivative_1], name='sigmoid_derivative')
        sigmoid_derivative_by_action = tf.multiply(sigmoid_derivative,action)
        sigmoid_derivative_by_action = tf.reduce_sum(
        sigmoid_derivative_by_action,axis=1,keep_dims=True)
        log_gradient_weights = tf.multiply(sigmoid_derivative_by_action, observation)

        ## could use compute_gradients to calculate gradients automatically
        # logGradient0 = tf.train.Optimizer\
        # .compute_gradients(tf.log(action_prob),weights_a1)
        # logGradient1 = tf.train.Optimizer\
        # .compute_gradients(tf.log(1-action_prob),weights_a1)
        # logGradients = tf.concat(1,[logGradient0,logGradient1])
        # logGradient = tf.multiply()

        #log_gradient_biases = sigmoid_derivative_by_action
        advantage = target - value
        gradient = tf.reduce_mean(tf.multiply(log_gradient_weights,advantage),axis=0)
        train_action = tf.train.AdamOptimizer(learning_rate=0.001)\
        .apply_gradients([(gradient,weights_a1)])
