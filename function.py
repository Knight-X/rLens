import tensorflow as tf
import backend as bk

class Function:
    def __init__(self):
        self._name = "go" 
        self._ops = [] 
        self._inputs = []
        self._outputs = []
        self._updateops = [] 
        self._backend = bk.TFBackend("go")
    def run_init(self):
        self._backend.init()

    def createPred(self):
        return;

    def createOptimizer(self):
        return;

    def run(self, params):
        return;

    def update(self):
        return self._backend.runOptimizer(self._update_ops)


def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    with tf.variable_scope(scope):
        dense = input_placeholder 
        conv1 = tf.layers.conv2d(inputs=dense,
                         filters=32,
                         kernel_size=[5, 5],
                         padding="same",
                         activation=tf.nn.relu
                        )
        max1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2) 
        conv2 = tf.layers.conv2d(inputs=max1,
                         filters=64,
                         kernel_size=[5, 5],
                         padding="same",
                         activation=tf.nn.relu
                        )
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2) 
        pool2_flat = tf.reshape(pool2, [-1, 11 * 11 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        for i in range(n_layers):
            dense = tf.layers.dense(
                inputs=dense,
                units=size,
                activation=activation)
            size = size / 2
        return tf.layers.dense(
            inputs=dense,
            units=output_size,
            activation=output_activation), pool2

class ActorFunc(Function):
    def __init__(self):
        Function.__init__(self)
    
    def createPred(self, actionsize, n_layers, size):
        discrete = True
        sy_ob_no = tf.placeholder(shape=[None, actionsize , actionsize, 1], name="ob", dtype=tf.float32)
        if discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        if discrete:
                    # YOUR_CODE_HERE
            sy_logits_na, pool2 = build_mlp(
                input_placeholder=sy_ob_no,
                output_size=actionsize,
                scope="build_nn",
                n_layers=n_layers,
                size=size,
                activation=tf.nn.relu)

            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=[1]) # Hint: Use the tf.multinomial op
            sy_soft = tf.nn.softmax(sy_logits_na) # Hint: Use the tf.multinomial op
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sy_ac_na,
                logits=sy_logits_na)
        else:
            # YOUR_CODE_HERE
            sy_mean  = build_mlp(
                input_placeholder=sy_ob_no,
                output_size=ac_dim,
                scope="build_nn",
                n_layers=n_layers,
                size=size,
                activation=tf.nn.relu)
            sy_logstd = tf.get_variable("logstd",shape=[ac_dim]) # logstd should just be a trainable variable, not a network output.
            sy_sampled_ac = sy_mean + tf.multiply(tf.exp(sy_logstd), tf.random_normal(tf.shape(sy_mean)))
            dist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean,
                                                                   scale_diag=tf.exp(sy_logstd))
            sy_logprob_n = -dist.log_prob(sy_ac_na)
        
        self._inputs.append(sy_ob_no)
        self._inputs.append(sy_ac_na)
        self._inputs.append(sy_adv_n)
        self._outputs.append(sy_logprob_n)
        self._ops.append(sy_soft)
        self._ops.append(sy_sampled_ac)

    def run(self, params):
        return self._backend.runAction(self._ops, self._inputs[0], params)
    def createOptimizer(self, learning_rate):
        weighted_negative_likelihood = tf.multiply(self._outputs[0], self._inputs[2])
        loss = tf.reduce_mean(weighted_negative_likelihood)  # Loss function that we'll differentiate to get the policy gradient.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self._updateops.append(optimizer)
        self._updateops.append(loss)

    def update(self, ob_no, ac_na, adv_n):
        return self._backend.runOptimizer(self._updateops, self._inputs[0], self._inputs[1], self._inputs[2], ob_no, ac_na, adv_n)



