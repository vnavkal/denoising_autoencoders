from tensorflow.examples.tutorials import mnist


def load_train():
    return mnist.input_data.read_data_sets('tmp/').train


def load_test():
    return mnist.input_data.read_data_sets('tmp/').test
