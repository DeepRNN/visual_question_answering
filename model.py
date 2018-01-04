import math
import os
import tensorflow as tf
import numpy as np

from base_model import *
from utils.nn import *
from episodic_memory import *

class QuestionAnswerer(BaseModel):
    def build(self):
        """ Build the model. """
        with tf.variable_scope("CNN"):
            self.build_cnn()

        with tf.variable_scope("RNN"):
            self.build_rnn()

        if self.mode=="train":
            with tf.variable_scope("Summary"):
                self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN part...")
        if self.cnn_model=='vgg16':
            self.build_vgg16()
        elif self.cnn_model=='resnet50':
            self.build_resnet50()
        elif self.cnn_model=='resnet101':
            self.build_resnet101()
        else:
            self.build_resnet152()
        print("CNN part built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        use_batch_norm = self.params.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = nonlinear(conv1_1_feats, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = nonlinear(conv1_2_feats, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = nonlinear(conv2_1_feats, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = nonlinear(conv2_2_feats, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = nonlinear(conv3_1_feats, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = nonlinear(conv3_2_feats, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = nonlinear(conv3_3_feats, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = nonlinear(conv4_1_feats, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = nonlinear(conv4_2_feats, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = nonlinear(conv4_3_feats, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = nonlinear(conv5_1_feats, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = nonlinear(conv5_2_feats,  'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = nonlinear(conv5_3_feats, 'relu')

        self.permutation = self.get_permutation(14, 14)
        conv5_3_feats.set_shape([self.batch_size, 14, 14, 512])
        conv5_3_feats_flat = self.flatten_feats(conv5_3_feats, 512)
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [196, 512]

        self.imgs = imgs
        self.is_train = is_train

    def basic_block(self, input_feats, name1, name2, is_train, use_batch_norm, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, use_batch_norm)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, use_batch_norm)
        branch2a_feats = nonlinear(branch2a_feats, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, use_batch_norm)
        branch2b_feats = nonlinear(branch2b_feats, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, use_batch_norm)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, use_batch_norm, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, use_batch_norm)
        branch2a_feats = nonlinear(branch2a_feats, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, use_batch_norm)
        branch2b_feats = nonlinear(branch2b_feats, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, use_batch_norm)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_resnet50(self):
        """ Build the ResNet50 net. """
        use_batch_norm = self.params.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)     

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, use_batch_norm, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, use_batch_norm, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, use_batch_norm, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, use_batch_norm, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, use_batch_norm, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, use_batch_norm, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, use_batch_norm, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, use_batch_norm, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet101(self):
        """ Build the ResNet101 net. """
        use_batch_norm = self.params.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, use_batch_norm, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, use_batch_norm, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet152(self):
        """ Build the ResNet152 net. """
        use_batch_norm = self.params.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, use_batch_norm, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, use_batch_norm, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats.set_shape([self.batch_size, 7, 7, 2048])
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.img_files = img_files
        self.is_train = is_train

    def get_permutation(self, height, width):
        """ Get the permutation corresponding to a snake-like walk as decribed by the paper. Used to flatten the convolutional feats. """
        permutation = np.zeros(height*width, np.int32)
        for i in range(height):
            for j in range(width):
                permutation[i*width+j] = i*width+j if i%2==0 else (i+1)*width-j-1
        return permutation

    def flatten_feats(self, feats, channels):
        """ Flatten the feats. """
        temp1 = tf.reshape(feats, [self.batch_size, -1, channels])
        temp1 = tf.transpose(temp1, [1, 0, 2])
        temp2 = tf.gather(temp1, self.permutation)
        temp2 = tf.transpose(temp2, [1, 0, 2])
        return temp2

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN part...")
        params = self.params
        use_batch_norm = params.use_batch_norm      
        fc_keep_prob = params.fc_keep_prob
        gru_keep_prob = params.gru_keep_prob
        is_train = self.is_train
        batch_size = self.batch_size                  

        dim_hidden = params.dim_hidden                     
        dim_embed = params.dim_embed                       
        max_ques_len = params.max_ques_len                 

        num_facts = self.conv_feat_shape[0]                                      
        dim_fact = self.conv_feat_shape[1]                                      
        num_words = self.word_table.num_words              

        facts = self.conv_feats

        questions = tf.placeholder(tf.int32, [batch_size, max_ques_len])        
        question_lens = tf.placeholder(tf.int32, [batch_size])                   
        answers = tf.placeholder(tf.int32, [batch_size])                        
        answer_weights = tf.placeholder(tf.float32, [batch_size])                        
        
        # Initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)])  
        emb_w = weight('emb_weights', [num_words, dim_embed], init_val=idx2vec)                

        # Encode the questions
        with tf.variable_scope('Question'):
            gru = tf.nn.rnn_cell.GRUCell(dim_hidden)
            gru = tf.nn.rnn_cell.DropoutWrapper(gru, gru_keep_prob, gru_keep_prob, gru_keep_prob)

            word_list = tf.unstack(questions, axis=1)                                             
            ques_embed = [tf.nn.embedding_lookup(emb_w, word) for word in word_list]             
            ques_embed = tf.transpose(tf.stack(ques_embed, axis=0), [1, 0, 2])   

            with tf.variable_scope('gru') as scope:
                all_states, final_state = tf.nn.dynamic_rnn(gru, ques_embed, dtype=tf.float32)       

            question_enc = []
            for k in range(batch_size):
                current_ques_enc = tf.slice(all_states, [k, question_lens[k]-1, 0], [1, 1, dim_hidden]) 
                question_enc.append(tf.squeeze(current_ques_enc))

            question_enc = tf.stack(question_enc, axis=0)                                                 

        # Encode the facts
        with tf.name_scope('InputFusion'):
            gru = tf.nn.rnn_cell.GRUCell(dim_hidden)
            gru = tf.nn.rnn_cell.DropoutWrapper(gru, gru_keep_prob, gru_keep_prob, gru_keep_prob)


            with tf.variable_scope('Forward'):
                with tf.variable_scope('gru') as scope:
                    forward_states, _ = tf.nn.dynamic_rnn(gru, facts, dtype=tf.float32)           

            with tf.variable_scope('Backward'):
                reversed_facts = tf.reverse(facts, [1])                   
                with tf.variable_scope('gru') as scope:
                    backward_states, _ = tf.nn.dynamic_rnn(gru, reversed_facts, dtype=tf.float32) 
                backward_states = tf.reverse(backward_states, [1])              

            facts_enc = forward_states + backward_states                                      

        # Episodic Memory Update
        with tf.variable_scope('EpisodicMemory'):
            episode = EpisodicMemory(dim_hidden, num_facts, question_enc, facts_enc, params.attention, is_train, use_batch_norm)
            memory = tf.identity(question_enc)                                                   
            gru = tf.nn.rnn_cell.GRUCell(dim_hidden)
            gru = tf.nn.rnn_cell.DropoutWrapper(gru, gru_keep_prob, gru_keep_prob, gru_keep_prob)

            # Tied memory weights
            if params.tie_memory_weight: 
                with tf.variable_scope('Layer') as scope:
                    for t in range(params.memory_step):
                        if params.memory_update == 'gru': 
                            with tf.variable_scope('gru') as scope:
                                memory = gru(episode.new_fact(memory), memory)[0]                     
                        else:
                            fact = episode.new_fact(memory)                                        
                            expanded_memory = tf.concat([memory, fact, question_enc], 1)           
                            memory = fully_connected(expanded_memory, dim_hidden, 'EM_fc')
                            memory = batch_norm(memory, 'EM_bn', is_train, use_batch_norm)
                            memory = nonlinear(memory, 'tanh')  
                        scope.reuse_variables()

            # Untied memory weights
            else:
                for t in range(params.memory_step):
                    with tf.variable_scope('Layer%d' %t) as scope:
                        if params.memory_update == 'gru':
                            with tf.variable_scope('gru') as scope:
                                memory = gru(episode.new_fact(memory), memory)[0]                     
                        else:
                            fact = episode.new_fact(memory)                                        
                            expanded_memory = tf.concat([memory, fact, question_enc], 1)           
                            memory = fully_connected(expanded_memory, dim_hidden, 'EM_fc')
                            memory = batch_norm(memory, 'EM_bn', is_train, use_batch_norm)
                            memory = nonlinear(memory, 'tanh')  

        memory = dropout(memory, fc_keep_prob, is_train)                                                  
        
        # Compute the result
        with tf.variable_scope('Result'):    
            expanded_memory = tf.concat([memory, question_enc], 1)    
            logits = fully_connected(expanded_memory, num_words, 'dec')
            results = tf.argmax(logits, 1)                                                        
            all_probs = tf.nn.softmax(logits)                                                    
            probs = tf.reduce_max(all_probs, 1)                                                      

        # Compute the loss
        with tf.variable_scope('Loss'):        
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers, logits=logits) 
            cross_entropy_loss = cross_entropy * answer_weights
            cross_entropy_loss = tf.reduce_sum(cross_entropy_loss) / tf.reduce_sum(answer_weights)

            if self.train_cnn:
                g_vars = tf.trainable_variables()
            else:
                g_vars = [tf_var for tf_var in tf.trainable_variables() if "CNN" not in tf_var.name]

            l2_loss = params.weight_decay * sum(tf.nn.l2_loss(tf_var) for tf_var in g_vars 
                                                                      if ("bias" not in tf_var.name and
                                                                          "offset" not in tf_var.name and 
                                                                          "scale" not in tf_var.name)) 
            loss = cross_entropy_loss + l2_loss

        # Build the solver
        with tf.variable_scope("Solver", reuse=tf.AUTO_REUSE):
            learning_rate = tf.train.exponential_decay(params.learning_rate, 
                                                   self.global_step,
                                                   10000, 
                                                   0.95, 
                                                   staircase=True)

            if params.solver=="momentum":
                solver = tf.train.MomentumOptimizer(learning_rate, params.momentum)
            elif params.solver=="rmsprop":
                solver = tf.train.RMSPropOptimizer(learning_rate, params.decay, params.momentum)
            else:
                solver = tf.train.GradientDescentOptimizer(learning_rate)

            gs = tf.gradients(loss, g_vars)
            gs, _ = tf.clip_by_global_norm(gs, 5.0)
            opt_op = solver.apply_gradients(zip(gs, g_vars), global_step=self.global_step)

        self.questions = questions
        self.question_lens = question_lens
        self.answers = answers
        self.answer_weights = answer_weights

        self.loss = loss
        self.cross_entropy_loss = cross_entropy_loss
        self.l2_loss = l2_loss

        self.opt_op = opt_op
        self.g_vars = g_vars
        self.gs = gs
        
        self.results = results
        self.probs = probs
        
        print("RNN part built.")        

    def build_summary(self):
        """Build the summary (for TensorBoard visualization)"""
        assert self.mode=="train"
        with tf.name_scope("Variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("Gradients"):
            for g, var in zip(self.gs, self.g_vars):
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(g)

        with tf.name_scope("Losses"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("l2_loss", self.l2_loss)
            tf.summary.scalar("loss", self.loss)
       
        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def get_feed_dict(self, batch, is_train):
        """ Get the feed dictionary for the current batch. """
        if is_train:
            # training phase
            img_files, questions, question_lens, answers = batch
            imgs = self.img_loader.load_imgs(img_files)
            answer_weights = self.word_weight[answers]
            return {self.imgs: imgs, 
                    self.questions: questions, 
                    self.question_lens: question_lens, 
                    self.answers: answers, 
                    self.answer_weights: answer_weights,  
                    self.is_train: is_train}
        else:
            # testing or validation phase
            img_files, questions, question_lens = batch
            imgs = self.img_loader.load_imgs(img_files)
            return {self.imgs: imgs, 
                    self.questions: questions, 
                    self.question_lens: question_lens, 
                    self.is_train: is_train} 

