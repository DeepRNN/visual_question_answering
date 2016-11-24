import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from dataset import *
from utils.words import *
from utils.vqa.vqa import *
from utils.vqa.vqaEval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True 
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, img_file):      
        """ Load and preprocess an image. """
        img = cv2.imread(img_file)

        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)

        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs


class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1

        self.cnn_model = params.cnn_model
        self.train_cnn = params.train_cnn

        self.save_dir = os.path.join(params.save_dir, self.cnn_model+'/')
        self.class_balancing_factor = params.class_balancing_factor

        self.img_loader = ImageLoader(params.mean_file)
        self.img_shape = [224, 224, 3]

        self.word_table = WordTable(params.dim_embed, params.max_ques_len, params.word_table_file)
        self.word_table.load()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build()
        self.saver = tf.train.Saver(max_to_keep=100)

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train, feats=None):
        raise NotImplementedError()

    def train(self, sess, train_vqa, train_data):
        """ Train the model. """
        print("Training the model...")
        params = self.params
        num_epochs = params.num_epochs

        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()

                if self.train_cnn:
                    # Train CNN and RNN 
                    feed_dict = self.get_feed_dict(batch, is_train=True)
                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)
 
                else:              
                    # Train RNN only 
                    img_files, _, _, _ = batch
                    imgs = self.img_loader.load_imgs(img_files)
                    feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=True, feats=feats)
                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)

                print(" Loss0=%f Loss1=%f" %(loss0, loss1))

                if (global_step + 1) % params.save_period == 0:
                    self.save(sess)

            train_data.reset()

        print("Training complete.")

    def val(self, sess, val_vqa, val_data, show_result=False):
        """ Validate the model. """
        print("Validating the model...")
        answers = []
        result_dir = self.params.val_result_dir

        # Compute the answers to the questions
        for k in tqdm(list(range(val_data.count))):
            batch = val_data.next_batch()
            img_files, questions, question_lens = batch

            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]

            if self.train_cnn: 
                feed_dict = self.get_feed_dict(batch, is_train=False) 
            else: 
                img_files, _, _ = batch 
                imgs = self.img_loader.load_imgs(img_files)
                feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False}) 
                feed_dict = self.get_feed_dict(batch, is_train=False, feats=feats)

            result = sess.run(self.results, feed_dict=feed_dict)
            answer = self.word_table.idx2word[result.squeeze()]
            answers.append({'question_id': val_data.question_ids[k], 'answer': answer})

            # Save the result in an image file
            question = questions[0]
            q_len = question_lens[0]
            q_words = ['Q:'] + [self.word_table.idx2word[question[i]] for i in range(q_len)]
            if q_words[-1]!='?':
                q_words.append('?')
            ques = ' '.join(q_words)
            a_words = ['A:'] + [answer]
            ans = ' '.join(a_words)

            img = mpimg.imread(img_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(ques+'\n'+ans)
            plt.savefig(os.path.join(result_dir, img_name+'_'+str(val_data.question_ids[k])+'_result.jpg'))

        val_data.reset() 

        # Evaluate these answers
        val_res_vqa = val_vqa.loadRes2(answers)
        scorer = VQAEval(val_vqa, val_res_vqa)
        scorer.evaluate()
        print("Validation complete.")

    def test(self, sess, test_data):
        """ Test the model. """
        print("Testing the model...")
        font = cv2.FONT_HERSHEY_COMPLEX        
        test_info_file = self.params.test_info_file
        result_file = self.params.test_result_file
        result_dir = self.params.test_result_dir

        question_ids = []
        answers = []

        # Compute the answers to the questions        
        for k in tqdm(list(range(test_data.count))):
            batch = test_data.next_batch()
            img_files, questions, question_lens = batch

            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]

            if self.train_cnn: 
                feed_dict = self.get_feed_dict(batch, is_train=False) 
            else: 
                img_files, _, _ = batch 
                imgs = self.img_loader.load_imgs(img_files)
                feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False}) 
                feed_dict = self.get_feed_dict(batch, is_train=False, feats=feats)            

            result = sess.run(self.results, feed_dict=feed_dict)           
            answer = self.word_table.idx2word[result.squeeze()]
            answers.append(answer)
            question_ids.append(test_data.question_ids[k])

            # Save the result in an image file
            question = questions[0]
            q_len = question_lens[0]
            q_words = ['Q:'] + [self.word_table.idx2word[question[i]] for i in range(q_len)]
            if q_words[-1]!='?':
                q_words.append('?')
            ques = ' '.join(q_words)
            a_words = ['A:'] + [answer]
            ans = ' '.join(a_words)

            img = mpimg.imread(img_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(ques+'\n'+ans)
            plt.savefig(os.path.join(result_dir, img_name+'_'+str(test_data.question_ids[k])+'_result.jpg'))

        # Save the answers to a file
        test_info = pd.read_csv(test_info_file)
        results = pd.DataFrame({'question_id': question_ids, 'answer': answers})
        results = pd.merge(test_info, results)
        results.to_csv(result_file)
        print("Testing complete.")

    def save(self, sess):
        """ Save the model. """
        print(("Saving model to %s" %self.save_dir))
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        """ Load the model. """
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def load2(self, data_path, session, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading CNN model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                       #print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError:
                        miss_count += 1
                       #print("Variable %s:%s missed" %(op_name, param_name))
                        if not ignore_missing:
                            raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))

