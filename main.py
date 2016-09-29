#!/usr/bin/env python
import os
import sys
import argparse
import tensorflow as tf

from model import *
from utils.dataset import *
from utils.vqa.vqa import *

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phase', default = 'train')
    parser.add_argument('--load', action = 'store_true', default = False)

    parser.add_argument('--mean_file', default = './utils/ilsvrc_2012_mean.npy')
    parser.add_argument('--cnn_model', default = 'vgg16')
    parser.add_argument('--cnn_model_file', default = './tfmodels/vgg16.tfmodel')
    parser.add_argument('--load_cnn_model', action = 'store_true', default = False)
  
    parser.add_argument('--train_image_dir', default = './train/images/')
    parser.add_argument('--train_question_file', default = './train/OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--train_answer_file', default = './train/mscoco_train2014_annotations.json')
    parser.add_argument('--train_annotation_file', default = './train/anns.csv')

    parser.add_argument('--val_image_dir', default = './val/images/')
    parser.add_argument('--val_question_file', default = './val/OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--val_answer_file', default = './val/mscoco_val2014_annotations.json')
    parser.add_argument('--val_annotation_file', default = './val/anns.csv')

    parser.add_argument('--test_image_dir', default = './test/images/')
    parser.add_argument('--test_question_file', default = './test/questions.csv')
    parser.add_argument('--test_info_file', default = './test/info.csv')
    parser.add_argument('--test_result_file', default = './test/results.csv')

    parser.add_argument('--word_table_file', default = './words/word_table.pickle')
    parser.add_argument('--glove_dir', default = './words/')
    parser.add_argument('--word2vec_scale', type = float, default = 0.2)
    parser.add_argument('--max_ques_len', type = int, default = 30)

    parser.add_argument('--save_dir', default = './models/')
    parser.add_argument('--save_period', type = int, default = 1)

    parser.add_argument('--solver', default = 'adam')     
    parser.add_argument('--num_epochs', type = int, default = 100) 
    parser.add_argument('--batch_size', type = int, default = 32) 
    parser.add_argument('--learning_rate', type = float, default = 5e-4) 
    parser.add_argument('--weight_decay', type = float, default = 1e-4) 
    parser.add_argument('--momentum', type = float, default = 0.9) 
    parser.add_argument('--decay', type = float, default = 0.9) 
    parser.add_argument('--batch_norm', action = 'store_true', default = False)

    parser.add_argument('--dim_hidden', type = int, default = 200)
    parser.add_argument('--dim_embed', type = int, default = 300)
    parser.add_argument('--init_embed_weight', action = 'store_true', default = False)
    parser.add_argument('--fix_embed_weight', action = 'store_true', default = False)
    parser.add_argument('--train_cnn', action = 'store_true', default = False)

    parser.add_argument('--memory_step', type = int, default = 3)
    parser.add_argument('--memory_update', default = 'gru')
    parser.add_argument('--attention', default = 'soft')
    parser.add_argument('--tie_memory_weight', action = 'store_true', default = False)

    args = parser.parse_args()

    with tf.Session() as sess:
        if args.phase == 'train':
            train_vqa, train_data = prepare_train_data(args)

            model = QuestionAnswerer(args, 'train')
            sess.run(tf.initialize_all_variables())

            if args.load:
                model.load(sess)
            elif args.load_cnn_model:
                model.load2(args.cnn_model_file, sess)

            model.train(sess, train_vqa, train_data)

        elif args.phase == 'val':
            val_vqa, val_data = prepare_val_data(args)
            model = QuestionAnswerer(args, 'val')
            model.load(sess)
            model.val(sess, val_vqa, val_data)

        else:
            test_data = prepare_test_data(args)
            model = QuestionAnswerer(args, 'test')          
            model.load(sess)
            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

