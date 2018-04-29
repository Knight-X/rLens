import tensorflow as tf
def createFunc_helper():
    return ActorFunc()

class NNBackend:
    def __init__(self):
        self._func = None

    def runAction(self):
        return;

    def runOptimizer(self):
        return;

class TFBackend(NNBackend):
    def __init__(self, name):
        tf.set_random_seed(0)

    def init(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        self._sess = tf.Session(config=tf_config)
        self._sess.__enter__() # equivalent to `with sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def runAction(self, ops, sy_ob_no, params):
        return self._sess.run(ops, feed_dict={sy_ob_no: params})

    def runOptimizer(self, updateops, sy_ob_no, sy_ac_na, sy_adv_n, ob_no, ac_na, adv_n):
        return self._sess.run(updateops, feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})


