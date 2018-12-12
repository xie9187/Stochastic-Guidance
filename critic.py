# ===========================
#   Critic DNN
# ===========================
import tensorflow as tf
import numpy as np
# Network Parameters - Hidden layers
n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 100

def reinforce(x, reward, n_hidden=500, scope=None):

  with tf.variable_scope(scope or 'REINFORCE_layer'):
    # important: stop the gradients
    x = tf.stop_gradient(x)
    reward = tf.stop_gradient(reward)
    # baseline: central
    # init = tf.constant(2.9428)
    init = tf.constant(0.)
    baseline_c = tf.get_variable('baseline_c', initializer=init)
    # baseline: data dependent
    baseline_x = (linear(
        tf.sigmoid(
            linear(
                tf.sigmoid(linear(
                    x, n_hidden, True, scope='l1')),
                n_hidden,
                True,
                scope='l2')),
            1,
            True,
            scope='l3'))

    reward = reward - baseline_c - baseline_x
    # reward = reward - baseline_x

    return reward

def linear(inputs,
            output_size,
            bias,
            bias_start_zero=False,
            matrix_start_zero=False,
            scope=None):
  """Define a linear connection that can customise the parameters."""

  shape = inputs.get_shape().as_list()

  if len(shape) != 2:
    raise ValueError('Linear is expecting 2D arguments: %s' % str(shape))
  if not shape[1]:
    raise ValueError('Linear expects shape[1] of arguments: %s' % str(shape))
  input_size = shape[1]

  # Now the computation.
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix = tf.get_variable(
          'Matrix', [input_size, output_size],
          initializer=tf.constant_initializer(0))
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size])
    res = tf.matmul(inputs, matrix)
    if not bias:
      return res
    if bias_start_zero:
      bias_term = tf.get_variable(
          'Bias', [output_size], initializer=tf.constant_initializer(0))
    else:
      bias_term = tf.get_variable('Bias', [output_size])
  return res + bias_term

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name="Weight")

def bias_variable(shape, val=0.03):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial, name="Bias")

def weight_variable_uniform(shape, value):
    initial = tf.random_uniform(shape, minval=-value, maxval=value, dtype=tf.float32)
    return tf.Variable(initial, name="Weights")

def bias_variable_uniform(shape, value):
    initial = tf.random_uniform(shape, minval=-value, maxval=value, dtype=tf.float32)
    return tf.Variable(initial, name="Bias")

def conv1d(x, W, stride):
    return tf.nn.conv1d(x, W, stride = stride, padding = "SAME")

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, switch_dim, learning_rate, tau, num_actor_vars, \
                baseline_rate=1., control_variance_flag=True):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.switch_dim = switch_dim
        self.learning_rate = learning_rate
        self.tau = tau
        # Create the critic network
        with tf.name_scope("OnlineNet"):
            self.inputs, self.action, self.out, self.logits, self.switch_a  = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.name_scope("TargetNet"):
            self.target_inputs, self.target_action, self.target_out, self.target_logits, self.target_switch_a = \
                self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        self.episode_r = tf.placeholder(tf.float32, shape=(), name='EpiR')
        self.episode_switch = tf.placeholder(tf.int32, shape=[None], name='EpiSwitch')

        # Define switch loss
        sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.episode_switch)
        if control_variance_flag:
            # with control variance
            reward_selection = tf.squeeze(reinforce(self.inputs, self.episode_r))
            self.switch_loss = tf.multiply(sample_loss, tf.stop_gradient(reward_selection))
            self.baseline_loss = tf.square(reward_selection)
            loss = self.switch_loss+self.baseline_loss*baseline_rate
            self.switch_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        else:
            # without control variance
            self.switch_loss = tf.multiply(sample_loss, self.episode_r)
            self.switch_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.switch_loss)

        # Define loss and optimization Op
        self.abs_errors = tf.reduce_sum(tf.abs(self.predicted_q_value - self.out), axis=1)
        self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.predicted_q_value, self.out))
        self.critic_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        # using Q value
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        # # FC1
        # with tf.name_scope("FC1"):
        #     w_fc1 = weight_variable([self.s_dim, n_hidden_1])
        #     b_fc1 = bias_variable([n_hidden_1])
        # # FC2
        # with tf.name_scope("FC2"):
        #     w_fc2 = weight_variable([n_hidden_1 + self.a_dim, n_hidden_2])
        #     b_fc2 = bias_variable([n_hidden_2])
        # # FC2 adv
        # with tf.name_scope("FC_adv"):
        #     w_fc2_adv = weight_variable([n_hidden_1, n_hidden_2/2])
        #     b_fc2_adv = bias_variable([n_hidden_2/2])
        # # FC2 critic
        # with tf.name_scope("FC_critic"):
        #     w_fc2_value = weight_variable([n_hidden_1, n_hidden_2/2])
        #     b_fc2_value = bias_variable([n_hidden_2/2])
        # # Out
        # with tf.name_scope("Out"):
        #     w_out = weight_variable([n_hidden_2, 1])
        #     b_out = bias_variable([1])
        # # Out_adv
        # with tf.name_scope("Out_adv"):
        #     w_out_adv = weight_variable([n_hidden_2/2, self.switch_dim])
        #     b_out_adv = bias_variable([self.switch_dim])
        # # Out_value
        # with tf.name_scope("Out_value"):
        #     w_out_value = weight_variable([n_hidden_2/2, 1])
        #     b_out_value = bias_variable([1])

        # FC1
        with tf.name_scope("FC1"):
            w_fc1 = weight_variable_uniform([self.s_dim, n_hidden_1], tf.sqrt(1.0/(self.s_dim)))
            b_fc1 = bias_variable_uniform([n_hidden_1], tf.sqrt(1.0/(self.s_dim)))
        # FC2
        with tf.name_scope("FC2"):
            w_fc2 = weight_variable_uniform([n_hidden_1+self.a_dim, n_hidden_2], tf.sqrt(1.0/(n_hidden_1+self.a_dim)))
            b_fc2 = bias_variable_uniform([n_hidden_2], tf.sqrt(1.0/(n_hidden_1+self.a_dim)))
        # FC2 adv
        with tf.name_scope("FC_adv"):
            w_fc2_adv = weight_variable_uniform([n_hidden_1, n_hidden_2/2], tf.sqrt(1.0/(n_hidden_1)))
            b_fc2_adv = bias_variable_uniform([n_hidden_2/2], tf.sqrt(1.0/(n_hidden_1)))
        # FC2 critic
        with tf.name_scope("FC_critic"):
            w_fc2_value = weight_variable_uniform([n_hidden_1, n_hidden_2/2], tf.sqrt(1.0/(n_hidden_1)))
            b_fc2_value = bias_variable_uniform([n_hidden_2/2], tf.sqrt(1.0/(n_hidden_1)))        
        # Out
        with tf.name_scope("Out"):
            w_out = weight_variable_uniform([n_hidden_2, 1], 3e-3)
            b_out = bias_variable_uniform([1], 3e-3)
        # Out_adv
        with tf.name_scope("Out_adv"):
            # w_out_adv = weight_variable_uniform([n_hidden_2/2, self.switch_dim-1], 3e-3)
            # b_out_adv = bias_variable_uniform([self.switch_dim-1], 3e-3)
            w_out_adv = weight_variable_uniform([n_hidden_2/2, self.switch_dim-1], 3e-3)
            b_out_adv = bias_variable_uniform([self.switch_dim-1], 3e-3)
                    # Out_adv
        with tf.name_scope("Out_value"):
            w_out_value = weight_variable_uniform([n_hidden_2/2, 1], 3e-3)
            b_out_value = bias_variable_uniform([1], 3e-3)

        # critic
        h_fc1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)
        h_fc1_a = tf.concat([h_fc1, action], axis=1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_a, w_fc2) + b_fc2)
        out = tf.matmul(h_fc2, w_out) + b_out

        # switch
        h_fc2_adv = tf.nn.relu(tf.matmul(h_fc1, w_fc2_adv) + b_fc2_adv) 
        h_fc2_value = tf.nn.relu(tf.matmul(h_fc1, w_fc2_value) + b_fc2_value)
        out_adv = tf.matmul(h_fc2_adv, w_out_adv) + b_out_adv
        out_value = tf.matmul(h_fc2_value, w_out_value) + b_out_value

        advAvg = tf.expand_dims(tf.reduce_mean(out_adv, axis=1), axis=1)
        advIdentifiable = tf.subtract(out_adv, advAvg)
        unscaled_logits = tf.add(out_value, advIdentifiable)
        probs = tf.sigmoid(unscaled_logits)
        [prob1, prob2] = tf.unstack(probs, axis=1)
        logit1 = tf.log(prob1)
        logit2 = tf.log((1-prob1)*prob2)
        logit3 = tf.log((1-prob1)*(1-prob2))
        logits = tf.stack([logit1, logit2, logit3], axis=1)

        # [prob1, temp_prob2, temp_prob3] = tf.unstack(probs, axis=1)
        # prob2 = temp_prob2/(temp_prob2+temp_prob3)
        # prob3 = temp_prob3/(temp_prob2+temp_prob3)
        # logit1 = tf.log(prob1)
        # logit2 = tf.log((1-prob1)*prob2)
        # logit3 = tf.log((1-prob1)*prob3)
        # logits = tf.stack([logit1, logit2, logit3], axis=1)
        
        switch_a = tf.multinomial(logits, 1)
        switch_a = tf.reshape(switch_a, [-1])

        return inputs, action, out, logits, switch_a

    def train(self, inputs, action, predicted_q_value, ISWeights):
        return self.sess.run([self.out, self.abs_errors, self.critic_optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.ISWeights: ISWeights
        })

    def switch_train(self, ep_state, ep_switch, ep_r):
        return self.sess.run(self.switch_optimize, feed_dict={
            self.inputs: ep_state,
            self.episode_switch: ep_switch,
            self.episode_r: ep_r
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict_switch(self, inputs):
        return self.sess.run([self.logits, self.switch_a], feed_dict={
            self.inputs: inputs
        })
