#! /usr/bin/env python

import tensorflow as tf

from model import ContextualAttentionNetwork

# Model Parameters that could be inferred from data loading
MAX_INPUT_LEN = 20
MAX_OUTPUT_LEN = 25
VOCAB_SIZE = 500
NUM_TOPICS = 5

# Model Parameters
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 128
FILTER_SIZE = 3

# Training Parameters
BATCH_SIZE = 100
DROPOUT_KEEP_PROB = 0.5
NUM_EPOCHS = 100

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        context_attn = ContextualAttentionNetwork(MAX_INPUT_LEN, MAX_OUTPUT_LEN, VOCAB_SIZE, NUM_TOPICS)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(context_attn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, x_batch_lens, c_batch, y_batch, y_batch_lens):
            # A single training step
            feed_dict = {
                context_attn.x: x_batch,
                context_attn.x_lens: x_batch_lens,
                context_attn.c: c_batch,
                context_attn.y: y_batch,
                context_attn.y_lens: y_batch_lens,
                context_attn.dropout_keep_prob: DROPOUT_KEEP_PROB
            }

            _, step, loss = sess.run([train_op, global_step, context_attn.loss], feed_dict)
            print('step {}: loss = {:g}'.format(step, loss))


        def val_step(x_batch, x_batch_lens, c_batch, y_batch, y_batch_lens):
            # A single validation step
            feed_dict = {
                context_attn.x: x_batch,
                context_attn.x_lens: x_batch_lens,
                context_attn.c: c_batch,
                context_attn.y: y_batch,
                context_attn.y_lens: y_batch_lens,
                context_attn.dropout_keep_prob: 1.0
            }

            step, loss = sess.run([global_step, context_attn.loss], feed_dict)
            print('step {}: loss = {:g}'.format(step, loss))
