import tensorflow as tf


def fc(x, num_in, num_out, name, act_func=None, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name) as scope:

        # Создаем переменные для весов (weights) и их смещений (biases)
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  initializer=initializer)
        biases = tf.get_variable('biases', [num_out],
                                 initializer=initializer)

        # Матричное умножение весов и добавление их смещения
        net = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if act_func is not None:
            # Apply ReLu non linearity
            # активация функции ReLu нелинейной.
            # ReLu - функция активации под названием «выпрямитель».
            # ReLU имеет следующую формулу f(x) = max(0, x) и реализует простой пороговый переход в нуле.
            return act_func(net)
        return net