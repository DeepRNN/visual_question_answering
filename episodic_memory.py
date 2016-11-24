import tensorflow as tf
from utils.nn import *

class AttnGRU:
    """ Attention-based GRU (used by the Episodic Memory Module). """
    def __init__(self, num_units, is_train, bn):
        self.num_units = num_units
        self.is_train = is_train
        self.bn = bn

    def __call__(self, inputs, state, attention):
        with tf.variable_scope('AttnGRU'):
            r = fully_connected(tf.concat(1, [inputs, state]), self.num_units, 'AttnGRU_fc1', init_b=1.0, group_id=1)
            r = batch_norm(r, 'AttnGRU_bn1', self.is_train, self.bn, 'sigmoid')

            c = fully_connected(tf.concat(1, [inputs, r*state]), self.num_units, 'AttnGRU_fc2', init_b=0.0, group_id=1)
            c = batch_norm(c, 'AttnGRU_bn2', self.is_train, self.bn, 'tanh') 

            new_state = attention * c + (1 - attention) * state
        return new_state


class EpisodicMemory:
    """ Episodic Memory Module. """
    def __init__(self, num_units, num_facts, question, facts, attention, is_train, bn):
        self.num_units = num_units                       
        self.num_facts = num_facts                           
        self.question = question                         
        self.facts = facts                         
        self.attention = attention
        self.is_train = is_train
        self.bn = bn
        self.attn_gru = AttnGRU(num_units, is_train, bn)

    def new_fact(self, memory):
        """ Get the context vector by using soft attention or attention-based GRU. """
        fact_list = tf.unpack(self.facts, axis=1)  
        mixed_fact = tf.zeros_like(fact_list[0])       

        atts = self.attend(memory)                       

        if self.attention=='gru':
            with tf.variable_scope('AttnGate') as scope:
                atts = tf.unpack(atts, axis=1)                                           
                for ctx, att in zip(fact_list, atts):
                    mixed_fact = self.attn_gru(ctx, mixed_fact, tf.expand_dims(att, 1))    
                    scope.reuse_variables()
        else:
            mixed_fact = tf.reduce_sum(self.facts * tf.expand_dims(atts, 2), 1)        
               
        return mixed_fact                                                                 

    def attend(self, memory):
        """ Get the attention weights. """
        c = self.facts                                                                
        q = tf.tile(tf.expand_dims(self.question, 1), [1, self.num_facts, 1])              
        m = tf.tile(tf.expand_dims(memory, 1), [1, self.num_facts, 1])                     

        z = tf.concat(2, [c * q, c * m, tf.abs(c - q), tf.abs(c - m)])                   
        z = tf.reshape(z, [-1, 4 * self.num_units])                                      

        z1 = fully_connected(z, self.num_units, 'EM_att_fc1', group_id=1)  
        z1 = batch_norm(z1, 'EM_att_bn1', self.is_train, self.bn, 'tanh')

        z2 = fully_connected(z1, 1, 'EM_att_fc2', group_id=1)
        z2 = batch_norm(z2, 'EM_att_bn2', self.is_train, self.bn, None)                
        z2 = tf.reshape(z2, [-1, self.num_facts])                                           

        atts = tf.nn.softmax(z2)                                                          
        return atts

