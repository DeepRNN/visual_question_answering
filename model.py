import tensorflow as tf
import numpy as np

from base_model import BaseModel
from episodic_memory import EpisodicMemory

class QuestionAnswerer(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn =='vgg16':
            self.build_vgg16()
        else:
            self.build_resnet50()
        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        self.permutation = self.get_permutation(14, 14)
        conv5_3_feats_flat = self.flatten_feats(conv5_3_feats, 512)
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [196, 512]
        self.images = images

    def build_resnet50(self):
        """ Build the ResNet50. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_feats = self.nn.conv2d(images,
                                  filters = 64,
                                  kernel_size = (7, 7),
                                  strides = (2, 2),
                                  activation = None,
                                  name = 'conv1')
        conv1_feats = self.nn.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.nn.max_pool2d(conv1_feats,
                                      pool_size = (3, 3),
                                      strides = (2, 2),
                                      name = 'pool1')

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)

        self.permutation = self.get_permutation(7, 7)
        res5c_feats_flat = self.flatten_feats(res5c_feats, 2048)
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]
        self.images = images

    def resnet_block(self, inputs, name1, name2, c, s=2):
        """ A basic block of ResNet. """
        branch1_feats = self.nn.conv2d(inputs,
                                    filters = 4*c,
                                    kernel_size = (1, 1),
                                    strides = (s, s),
                                    activation = None,
                                    use_bias = False,
                                    name = name1+'_branch1')
        branch1_feats = self.nn.batch_norm(branch1_feats, name2+'_branch1')

        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (s, s),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def resnet_block2(self, inputs, name1, name2, c):
        """ Another basic block of ResNet. """
        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a',)
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def get_permutation(self, height, width):
        """ Get the permutation corresponding to the snake-like walk decribed \
           in the paper. Used to flatten the convolutional feats. """
        permutation = np.zeros(height*width, np.int32)
        for i in range(height):
            for j in range(width):
                permutation[i*width+j] = i*width+j if i%2==0  \
                                         else (i+1)*width-j-1
        return permutation

    def flatten_feats(self, feats, channels):
        """ Flatten the feats. """
        temp1 = tf.reshape(feats, [self.config.batch_size, -1, channels])
        temp1 = tf.transpose(temp1, [1, 0, 2])
        temp2 = tf.gather(temp1, self.permutation)
        temp2 = tf.transpose(temp2, [1, 0, 2])
        return temp2

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        facts = self.conv_feats
        num_facts, dim_fact = self.conv_feat_shape

        # Setup the placeholders
        question_word_idxs = tf.placeholder(
            dtype = tf.int32,
            shape = [config.batch_size, config.max_question_length])
        question_lens = tf.placeholder(
            dtype = tf.int32,
            shape = [config.batch_size])
        if self.is_train:
            answer_idxs = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])
            if config.question_encoding == 'positional':
                position_weights = tf.placeholder(
                    dtype = tf.float32,
                    shape = [config.batch_size, \
                             config.max_question_length, \
                             config.dim_embedding])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Encode the questions
        with tf.variable_scope('question_encoding'):
            question_embeddings = tf.nn.embedding_lookup(
                embedding_matrix,
                question_word_idxs)

            if config.question_encoding == 'positional':
                # use positional encoding
                self.build_position_weights()
                question_encodings = question_embeddings * position_weights
                question_encodings = tf.reduce_sum(question_encodings,
                                                    axis = 1)
            else:
                # use GRU encoding
                outputs, _ = tf.nn.dynamic_rnn(
                    self.nn.gru(),
                    inputs = question_embeddings,
                    dtype = tf.float32)

                question_encodings = []
                for k in range(config.batch_size):
                    question_encoding = tf.slice(outputs,
                                                 [k, question_lens[k]-1, 0],
                                                 [1, 1, config.num_gru_units])
                    question_encodings.append(tf.squeeze(question_encoding))
                question_encodings = tf.stack(question_encodings, axis = 0)

        # Encode the facts
        with tf.variable_scope('input_fusion'):
            if config.embed_fact:
                facts = tf.reshape(facts, [-1, dim_fact])
                facts = self.nn.dropout(facts)
                facts = self.nn.dense(
                    facts,
                    units = config.dim_embedding,
                    activation = tf.tanh,
                    name = 'fc')
                facts = tf.reshape(facts, [-1, num_facts, config.dim_embedding])

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                self.nn.gru(),
                self.nn.gru(),
                inputs = facts,
                dtype = tf.float32)
            outputs_fw, outputs_bw = outputs
            fact_encodings = outputs_fw + outputs_bw

        # Episodic Memory Update
        with tf.variable_scope('episodic_memory'):
            episode = EpisodicMemory(config,
                                     num_facts,
                                     question_encodings,
                                     fact_encodings)
            memory = tf.identity(question_encodings)

            if config.tie_memory_weight:
                scope_list = ['layer'] * config.memory_step
            else:
                scope_list = ['layer'+str(t) for t in range(config.memory_step)]

            for t in range(config.memory_step):
                with tf.variable_scope(scope_list[t], reuse = tf.AUTO_REUSE):
                    fact = episode.new_fact(memory)
                    if config.memory_update == 'gru':
                        gru = self.nn.gru()
                        memory = gru(fact, memory)[0]
                    else:
                        expanded_memory = tf.concat(
                            [memory, fact, question_encodings],
                            axis = 1)
                        expanded_memory = self.nn.dropout(expanded_memory)
                        memory = self.nn.dense(
                            expanded_memory,
                            units = config.num_gru_units,
                            activation = tf.nn.relu,
                            name = 'fc')

        # Compute the result
        with tf.variable_scope('result'):
            expanded_memory = tf.concat([memory, question_encodings],
                                        axis = 1)
            expanded_memory = self.nn.dropout(expanded_memory)
            logits = self.nn.dense(expanded_memory,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'logits')
            prediction = tf.argmax(logits, axis = 1)

        # Compute the loss and accuracy if necessary
        if self.is_train:
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = answer_idxs,
                logits = logits)
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss + reg_loss

            ground_truth = tf.cast(answer_idxs, tf.int64)
            prediction_correct = tf.where(
                tf.equal(prediction, ground_truth),
                tf.cast(tf.ones_like(prediction), tf.float32),
                tf.cast(tf.zeros_like(prediction), tf.float32))
            accuracy = tf.reduce_mean(prediction_correct)

        self.question_word_idxs = question_word_idxs
        self.question_lens = question_lens
        self.prediction = prediction

        if self.is_train:
            self.answer_idxs = answer_idxs
            if config.question_encoding == 'positional':
                self.position_weights = position_weights
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy

        print("RNN built.")

    def build_position_weights(self):
        """ Setup the weights for the positional encoding of questions. """
        config = self.config
        D = config.dim_embedding
        pos_weights = []
        for M in range(config.max_question_length):
            cur_pos_weights = []
            for j in range(config.max_question_length):
                if j <= M:
                    temp = [1.0-(j+1.0)/(M+1.0) \
                            -((d+1.0)/D)*(1-2.0*(j+1.0)/(M+1.0)) \
                            for d in range(D)]
                else:
                    temp = [0.0] * D
                cur_pos_weights.append(temp)
            pos_weights.append(cur_pos_weights)
        self.pos_weights = np.array(pos_weights, np.float32)

    def build_optimizer(self):
        """ Setup the training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def get_feed_dict(self, batch):
        """ Get the feed dictionary for the current batch. """
        config = self.config
        if self.is_train:
            # training phase
            image_files, question_word_idxs, question_lens, answer_idxs = batch
            images = self.image_loader.load_images(image_files)
            if config.question_encoding == 'positional':
                position_weights = [self.pos_weights[question_lens[i]-1, :, :]
                                    for i in range(config.batch_size)]
                position_weights = np.array(position_weights, np.float32)
                return {self.images: images,
                        self.question_word_idxs: question_word_idxs,
                        self.question_lens: question_lens,
                        self.answer_idxs: answer_idxs,
                        self.position_weights: position_weights}
            else:
                return {self.images: images,
                        self.question_word_idxs: question_word_idxs,
                        self.question_lens: question_lens,
                        self.answer_idxs: answer_idxs}
        else:
            # evaluation or testing phase
            image_files, question_word_idxs, question_lens = batch
            images = self.image_loader.load_images(image_files)
            return {self.images: images,
                    self.question_word_idxs: question_word_idxs,
                    self.question_lens: question_lens}
