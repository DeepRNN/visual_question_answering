import tensorflow as tf
from utils.nn import NN

class AttnGRU(object):
    """ Attention-based GRU (used by the Episodic Memory Module). """
    def __init__(self, config):
        self.nn = NN(config)
        self.num_units = config.num_gru_units

    def __call__(self, inputs, state, attention):
        with tf.variable_scope('attn_gru'):
            r_input = tf.concat([inputs, state], axis = 1)
            r_input = self.nn.dropout(r_input)
            r = self.nn.dense(r_input,
                              units = self.num_units,
                              activation = None,
                              use_bias = False,
                              name = 'fc1')
            b = tf.get_variable('fc1/bias',
                                shape = [self.num_units],
                                initializer = tf.constant_initializer(1.0))
            r = tf.nn.bias_add(r, b)
            r = tf.sigmoid(r)

            c_input = tf.concat([inputs, r*state], axis = 1)
            c_input = self.nn.dropout(c_input)
            c = self.nn.dense(c_input,
                              units = self.num_units,
                              activation = tf.tanh,
                              name = 'fc2')

            new_state = attention * c + (1 - attention) * state
        return new_state

class EpisodicMemory(object):
    """ Episodic Memory Module. """
    def __init__(self, config, num_facts, question, facts):
        self.nn = NN(config)
        self.num_units = config.num_gru_units
        self.num_facts = num_facts
        self.question = question
        self.facts = facts
        self.attention = config.attention
        if self.attention == 'gru':
            self.attn_gru = AttnGRU(config)

    def new_fact(self, memory):
        """ Get the context vector by using either soft attention or
            attention-based GRU. """
        fact_list = tf.unstack(self.facts, axis = 1)
        mixed_fact = tf.zeros_like(fact_list[0])

        with tf.variable_scope('attend'):
            attentions = self.attend(memory)

        if self.attention == 'gru':
            with tf.variable_scope('attn_gate') as scope:
                attentions = tf.unstack(attentions, axis = 1)
                for ctx, att in zip(fact_list, attentions):
                    mixed_fact = self.attn_gru(ctx,
                                               mixed_fact,
                                               tf.expand_dims(att, 1))
                    scope.reuse_variables()
        else:
            mixed_fact = tf.reduce_sum(self.facts*tf.expand_dims(attentions, 2),
                                       axis = 1)

        return mixed_fact

    def attend(self, memory):
        """ Get the attention weights. """
        c = self.facts
        q = tf.tile(tf.expand_dims(self.question, 1), [1, self.num_facts, 1])
        m = tf.tile(tf.expand_dims(memory, 1), [1, self.num_facts, 1])

        z = tf.concat([c*q, c*m, tf.abs(c-q), tf.abs(c-m)], 2)
        z = tf.reshape(z, [-1, 4*self.num_units])

        z = self.nn.dropout(z)
        z1 = self.nn.dense(z,
                           units = self.num_units,
                           activation = tf.tanh,
                           name = 'fc1')
        z1 = self.nn.dropout(z1)
        z2 = self.nn.dense(z1,
                           units = 1,
                           activation = None,
                           use_bias = False,
                           name = 'fc2')
        z2 = tf.reshape(z2, [-1, self.num_facts])

        attentions = tf.nn.softmax(z2)
        return attentions
