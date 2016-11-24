#!/usr/bin/env python
import os
import sys
import argparse
import tensorflow as tf

from model import *
from dataset import *
from utils.vqa.vqa import *

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phase', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')

    parser.add_argument('--mean_file', default='./utils/ilsvrc_2012_mean.npy', help='Dataset image mean: a Numpy array with (Channel, Height, Width) dimensions')
    parser.add_argument('--cnn_model', default='vgg16', help='CNN model to use: Can be vgg16, resnet50, resnet101 or resnet152')
    parser.add_argument('--cnn_model_file', default='./tfmodels/vgg16.tfmodel', help='Tensorflow model file for the chosen CNN model')
    parser.add_argument('--load_cnn_model', action='store_true', default=False, help='Turn on to load the pretrained CNN model')
    parser.add_argument('--train_cnn', action='store_true', default=False, help='Turn on to jointly train CNN and RNN. Otherwise, only RNN is trained')
  
    parser.add_argument('--train_image_dir', default='./train/images/', help='Directory containing the COCO train2014 images')
    parser.add_argument('--train_question_file', default='./train/OpenEnded_mscoco_train2014_questions.json', help='JSON file storing the open-ended questions for COCO train2014 images')
    parser.add_argument('--train_answer_file', default='./train/mscoco_train2014_annotations.json', help='JSON file storing the answers to open-ended questions for COCO train2014 images')
    parser.add_argument('--train_annotation_file', default='./train/anns.csv', help='Temporary file to store the training information')

    parser.add_argument('--val_image_dir', default='./val/images/', help='Directory containing the COCO val2014 images')
    parser.add_argument('--val_question_file', default='./val/OpenEnded_mscoco_val2014_questions.json', help='JSON file storing the open-ended questions for COCO val2014 images')
    parser.add_argument('--val_answer_file', default='./val/mscoco_val2014_annotations.json', help='JSON file storing the answers to open-ended questions for COCO val2014 images')
    parser.add_argument('--val_annotation_file', default='./val/anns.csv', help='Temporary file to store the validation information')
    parser.add_argument('--val_result_dir', default='./val/results/', help='Directory to store the validation results as images')

    parser.add_argument('--test_image_dir', default='./test/images/', help='Directory containing the testing images')
    parser.add_argument('--test_question_file', default='./test/questions.csv', help='File storing the questions for testing images')
    parser.add_argument('--test_info_file', default='./test/info.csv', help='Temporary file to store the testing information')
    parser.add_argument('--test_result_file', default='./test/results.csv', help='File to store the testing results')
    parser.add_argument('--test_result_dir', default='./test/results/', help='Directory to store the testing results as images')

    parser.add_argument('--word_table_file', default='./words/word_table.pickle', help='Temporary file to store the word table')
    parser.add_argument('--glove_dir', default='./words/', help='Directory containing the GloVe data')
    parser.add_argument('--max_ques_len', type=int, default=30, help='Maximum length of the question. Extremely long questions will be ignored')

    parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model')
    parser.add_argument('--save_period', type=int, default=1000, help='Period to save the trained model')

    parser.add_argument('--solver', default='adam', help='Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)') 
    parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)') 
    parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization')  

    parser.add_argument('--dim_hidden', type=int, default=500, help='Dimension of the hidden state in each GRU')
    parser.add_argument('--dim_embed', type=int, default=300, help='Dimension of the word embedding')
    parser.add_argument('--memory_step', type=int, default=3, help='Number of memory update steps')
    parser.add_argument('--memory_update', default='gru', help='Memory update mechanism: Can be gru or mlp')
    parser.add_argument('--attention', default='soft', help='Attention mechanism: Can be soft or gru')
    parser.add_argument('--init_embed_with_glove', action='store_true', default=False, help='Turn on to initialize the word embedding with the GloVe data')  
    parser.add_argument('--fix_embed_weight', action='store_true', default=False, help='Turn on to fix the word embedding')
    parser.add_argument('--tie_memory_weight', action='store_true', default=False, help='Turn on to tie the memory weights at different time steps')
    parser.add_argument('--class_balancing_factor', type=float, default=0.0, help='Class balancing factor. The larger it is, the model pays more attention to rare words.') 

    args = parser.parse_args()

    with tf.Session() as sess:
        # training phase
        if args.phase == 'train':
            train_vqa, train_data = prepare_train_data(args)

            model = QuestionAnswerer(args, 'train')
            sess.run(tf.initialize_all_variables())

            if args.load:
                model.load(sess)
            elif args.load_cnn_model:
                model.load2(args.cnn_model_file, sess)

            model.train(sess, train_vqa, train_data)

        # validation phase
        elif args.phase == 'val':
            val_vqa, val_data = prepare_val_data(args)
            model = QuestionAnswerer(args, 'val')
            model.load(sess)
            model.val(sess, val_vqa, val_data)

        # testing phase
        else:
            test_data = prepare_test_data(args)
            model = QuestionAnswerer(args, 'test')          
            model.load(sess)
            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

