import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.words import *
from utils.dataset import *
from utils.vqa.vqa import *
from utils.vqa.vqaEval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.isotropic = False 
        self.channels = 3 
        self.bgr = True 
        self.scale_shape = [224, 224]
        self.crop_shape = [224, 224]
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, image_file):
        file_data = tf.read_file(image_file)

        img = tf.image.decode_jpeg(file_data, channels=self.channels)
        img = tf.reverse(img, [False, False, self.bgr]) 

        if self.isotropic:
            img_shape = tf.to_float(tf.shape(img)[:2])
            min_length = tf.minimum(img_shape[0], img_shape[1])
            scale_shape = tf.pack(self.scale_shape)
            new_shape = tf.to_int32((scale_shape / min_length) * img_shape)
        else:
            new_shape = tf.pack(self.scale_shape)

        img = tf.image.resize_images(img, new_shape[0], new_shape[1])

        crop_shape = tf.pack(self.crop_shape)
        offset = (new_shape - crop_shape) / 2     
        img = tf.slice(img, tf.to_int32([offset[0], offset[1], 0]), tf.to_int32([crop_shape[0], crop_shape[1], -1]))

        img = tf.to_float(img) - self.mean
        return img


class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1
        self.save_dir = params.save_dir

        self.cnn_model = params.cnn_model
        self.train_cnn = params.train_cnn

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
        print("Training the model ...")
        params = self.params
        num_epochs = params.num_epochs

        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):

                batch = train_data.next_batch()

                if self.train_cnn:
                    feed_dict = self.get_feed_dict(batch, is_train=True)
                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)
                else:              
                    img_files, _, _, _ = batch
                    feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=True, feats=feats)
                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)

                print(" Loss0=%f Loss1=%f" %(loss0, loss1))

            train_data.reset()

            if (epoch_no + 1) % params.save_period == 0:
                self.save(sess)

        print("Training complete.")

    def val(self, sess, val_vqa, val_data):
        print("Validating the model ...")
        answers = []

        for k in tqdm(list(range(val_data.count))):
            batch = val_data.next_batch()
            feed_dict = self.get_feed_dict(batch, is_train=False)
            result = sess.run(self.results, feed_dict=feed_dict)
            answer = self.word_table.idx2word[result]
            answers.append({'question_id': val_data.question_ids[k], 'answer': answer})

        val_data.reset() 

        val_res_vqa = val_vqa.loadRes2(answers)
        scorer = VQAEval(val_vqa, val_res_vqa)
        scorer.evaluate()
        print("Validation complete.")

    def test(self, sess, test_data):
        print("Testing the model ...")
        test_info_file = self.params.test_info_file
        test_result_file = self.params.test_result_file
        question_ids = []
        answers = []
        
        for k in tqdm(list(range(test_data.count))):
            batch = test_data.next_batch()
            feed_dict = self.get_feed_dict(batch, is_train=False)
            result = sess.run(self.results, feed_dict=feed_dict)
            answer = self.word_table.idx2word[result]
            answers.append(answer)
            question_ids.append(test_data.question_ids[k])

        test_info = pd.read_csv(test_info_file)
        results = pd.DataFrame({'question_id': question_ids, 'answer': answers})
        results = pd.merge(test_info, results)
        results.to_csv(test_result_file)
        print("Testing complete.")

    def save(self, sess):
        print(("Saving model to %s" %self.save_dir))
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def load2(self, data_path, session, ignore_missing=True):
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

