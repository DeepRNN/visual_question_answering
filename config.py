
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        self.max_question_length = 30
        self.dim_embedding = 512
        self.num_gru_units = 512
        self.memory_step = 3
        self.memory_update = 'relu'      # 'gru' or 'relu'
        self.attention = 'gru'           # 'gru' or 'soft'
        self.tie_memory_weight = False
        self.question_encoding = 'gru'   # 'gru' or 'positional'
        self.embed_fact = False

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-6
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-6
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.gru_drop_rate = 0.3

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 64
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 10000
        self.clip_gradients = 10.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-5

        # about the saver
        self.save_period = 1000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'

        # about the training
        self.train_image_dir = './train/images/'
        self.train_question_file = './train/OpenEnded_mscoco_train2014_questions.json'
        self.train_answer_file = './train/mscoco_train2014_annotations.json'
        self.temp_train_annotation_file = './train/anns.csv'
        self.temp_train_data_file = './train/data.npy'

        # about the evaluation
        self.eval_image_dir = './val/images/'
        self.eval_question_file = './val/OpenEnded_mscoco_val2014_questions.json'
        self.eval_answer_file = './val/mscoco_val2014_annotations.json'
        self.temp_eval_annotation_file = './val/anns.csv'
        self.temp_eval_data_file = './val/data.npy'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_question_file = './test/questions.csv'
        self.temp_test_info_file = './test/info.csv'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'
