import math
import time

import tensorflow as tf


def fit_and_evaluate(train, test, encoding_length, batch_size, corrupter, learning_rate, num_steps,
                     num_features):
    with tf.Graph().as_default():
        X_pl = tf.placeholder(tf.float32, shape=(None, num_features))

        corrupted = _corrupt(X_pl, corrupter)
        encoded, weights = _encode(corrupted, encoding_length, num_features)
        reconstruction = _decode(encoded, encoding_length, num_features)

        loss = _loss(X_pl, reconstruction)

        train_op = _training(loss, learning_rate)

        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)

        step = 0

        while step < num_steps:
            start_time = time.time()

            batch = train.next_batch(batch_size)[0]

            _, loss_value = sess.run((train_op, loss), feed_dict={X_pl: batch})

            duration = time.time() - start_time

            if step % 500 == 0:
                print(f'Step {step}: loss = {loss_value} ({duration} sec)')

            step += 1

        corrupted_val, reconstruction_val, weights_val = (
            sess.run((corrupted, reconstruction, weights), feed_dict={X_pl: test})
        )

    return corrupted_val, reconstruction_val, weights_val


def _corrupt(X, corrupter):
    return corrupter.corrupt(X)


def _encode(corrupted, encoding_length, num_features):
    with tf.name_scope('encoding'):
        weights = tf.Variable(
            tf.truncated_normal((num_features, encoding_length),
                                stddev=1/math.sqrt(num_features)),
            name='weights'
        )
    return tf.sigmoid(tf.matmul(corrupted, weights)), weights


def _decode(encoded, encoding_length, num_features):
    with tf.name_scope('decoding'):
        weights = tf.Variable(
            tf.truncated_normal((encoding_length, num_features),
                                stddev=1./math.sqrt(encoding_length)),
            name='weights'
        )
    return tf.matmul(encoded, weights)


def _loss(X_pl, reconstruction):
    return tf.nn.l2_loss(X_pl - reconstruction) / tf.cast(tf.shape(X_pl)[0], tf.float32)


def _training(loss, learning_rate):
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
