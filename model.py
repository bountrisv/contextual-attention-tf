import tensorflow as tf


class ContextualAttentionNetwork(object):
    """
    ContextualAttentionNetwork implements the graph for the "Contextual Attention Mechanism" from paper: "Neural
    Contextual Conversation Learning with Labeled Question-Answering Pairs", https://arxiv.org/pdf/1607.05809v1.pdf

    TensorFlow version: 0.12
    """

    def __init__(
            self, max_input_len, max_output_len, vocab_size, num_topics, embedding_size=256,
            hidden_size=128, filter_size=3):
        # Placeholders, None is batch size
        self.x = tf.placeholder(tf.int32, [None, max_input_len], name='x')
        self.y = tf.placeholder(tf.int32, [None, max_output_len], name='y')
        self.c = tf.placeholder(tf.float32, [None, num_topics], name='c')  # Context tensor computed by the CNN encoder
        self.x_lens = tf.placeholder(tf.int32, [None, ], name='x_lens')
        self.y_lens = tf.placeholder(tf.int32, [None, ], name='y_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.1)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Embedding Layer
        with tf.variable_scope('embedding'):
            w_embed = tf.get_variable('w_embed', [vocab_size, embedding_size], initializer=self.emb_initializer)
            self.embedded_encoder_input = tf.nn.embedding_lookup(w_embed, self.x)

        # Encoder RNN
        with tf.variable_scope('encoder'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell,
                self.embedded_encoder_input,
                sequence_length=self.x_lens,
                dtype=tf.float32)

        # "Gating" Outputs
        with tf.variable_scope('context-gate'):
            # Context gate calculation, expanded dims for broadcasting
            w_t_c = tf.get_variable('w_t_c', [max_input_len, num_topics], initializer=self.weight_initializer)
            w_t_h = tf.get_variable('w_t_h', [hidden_size, max_input_len, 1], initializer=self.weight_initializer)
            b_t_c = tf.get_variable('b_t_c', [max_input_len, 1], initializer=self.const_initializer)

            # Pointwise multiplications
            context_term = tf.matmul(w_t_c, self.c, transpose_b=True)
            outputs_term = tf.reduce_sum(tf.multiply(w_t_h, tf.transpose(self.encoder_outputs)), 0)
            context_gate = tf.sigmoid(context_term + outputs_term + b_t_c)  # [batch_size x max_input_len]

            # gated_outputs shape is [batch_size x max_input_len x hidden_size]
            self.gated_outputs = self.encoder_outputs * tf.expand_dims(tf.transpose(context_gate), -1)

        # Attention Layer
        with tf.variable_scope('attention'):
            # CNN layer to get attention vectors in every decoder input
            filter_shape = [filter_size, hidden_size, 1, max_output_len]
            w_conv = tf.get_variable('w_conv', filter_shape, initializer=self.weight_initializer)
            b_conv = tf.get_variable('b_conv', max_output_len, initializer=self.const_initializer)
            self.num_attn_features = max_input_len - filter_size + 1

            conv = tf.nn.conv2d(
                tf.expand_dims(self.gated_outputs, -1),
                w_conv, strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')
            # Attention vector (for each output) shape is: [batch_size x num_att_features x 1 x max_output_len]
            attn_vector = tf.nn.tanh(tf.nn.bias_add(conv, b_conv))
            attn_vector = tf.nn.dropout(attn_vector, self.dropout_keep_prob)

            # Reshape to rank 3 time major
            self.tim_maj_attn_vector = tf.transpose(attn_vector, perm=[3, 0, 1, 2])
            self.tim_maj_attn_vector = tf.reshape(
                self.tim_maj_attn_vector,
                shape=[max_output_len, -1, self.num_attn_features],
                name='tim_maj_attn_vector')

            # # If involving the context in decoder inputs concatenate it with attention vector to pass unto cell
            # wc_attn = tf.get_variable('wc_attn', [num_topics, max_output_len], initializer=self.weight_initializer)
            # decoder_input_c_term = tf.expand_dims(tf.transpose(tf.matmul(self.c, wc_attn)), -1)
            # self.tim_maj_attn_vector = decoder_input_c_term + self.tim_maj_attn_vector

        # RNN Decoder
        with tf.variable_scope('decoder'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)
            self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(
                cell,
                self.tim_maj_attn_vector,
                sequence_length=self.y_lens,
                initial_state=self.encoder_state,
                dtype=tf.float32,
                time_major=True)

        # Softmax Layer
        with tf.variable_scope('softmax'):
            w_softmax = tf.get_variable('w_softmax', [hidden_size, vocab_size], initializer=self.weight_initializer)
            b_softmax = tf.get_variable('b_softmax', [vocab_size], initializer=self.const_initializer)

            # # If involving the context in softmax
            # wc_softmax = tf.get_variable('wc_softmax', [num_topics, vocab_size], initializer=self.weight_initializer)
            # context_included = tf.matmul(self.c, wc_softmax)

            # Reshape to not time major outputs
            decoder_outputs_flat = tf.reshape(tf.transpose(self.decoder_outputs, perm=[1, 0, 2]), [-1, hidden_size])
            logits_flat = tf.matmul(decoder_outputs_flat, w_softmax) + b_softmax    # + context_included
            self.probs_flat = tf.nn.softmax(logits_flat)
            self.probs = tf.reshape(self.probs_flat, [-1, max_output_len, vocab_size], name='probs')

        # Loss Calculation
        with tf.variable_scope('loss'):
            y_flat = tf.reshape(self.y, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)  # Cross-entropy

            # Mask the loss (wrt zero padded outputs)
            mask = tf.sign(tf.to_float(y_flat))
            masked_losses = mask * losses
            masked_losses = tf.reshape(masked_losses, tf.shape(self.y))

            # Mean cross-entropy
            self.loss = tf.reduce_mean(tf.reduce_sum(masked_losses, reduction_indices=1), name='loss')
