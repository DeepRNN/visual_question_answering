import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle as pickle
from tqdm import tqdm
import json
import copy
import string

from utils.nn import NN
from utils.misc import ImageLoader
from utils.vqa.vqa import VQA
from utils.vqa.vqaEval import VQAEval

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.nn = NN(config)
        self.build()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the VQA training data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

        for epoch_no in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                feed_dict = self.get_feed_dict(batch)
                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict = feed_dict)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                train_writer.add_summary(summary, global_step)
            train_data.reset()

        print("Training complete.")

    def eval(self, sess, eval_gt_vqa, eval_data, vocabulary):
        """ Evaluate the model using the VQA validation data. """
        print("Evaluating the model...")
        config = self.config
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        question_ids = eval_data.question_ids
        answers = []

        # Compute the answers to the questions
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches))):
            batch = eval_data.next_batch()
            image_files, question_word_idxs, question_lens = batch
            feed_dict = self.get_feed_dict(batch)
            result = sess.run(self.prediction, feed_dict = feed_dict)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                answer = vocabulary.words[result[l]]
                answers.append(answer)

                # Save the result in an image file
                if config.save_eval_result_as_image:
                    image_file = image_files[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]

                    q_word_idxs = question_word_idxs[l]
                    q_len = question_lens[l]
                    q_words = [vocabulary.words[q_word_idxs[i]] \
                        for i in range(q_len)]
                    if q_words[-1] != '?':
                        q_words.append('?')
                    Q = 'Q: ' + ''.join([' '+w if not w.startswith("'") \
                        and w not in string.punctuation \
                        else w for w in q_words]).strip()
                    A = 'A: ' + answer

                    image = mpimg.imread(image_file)
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title(Q+'\n'+A)
                    plt.savefig(os.path.join(config.eval_result_dir, \
                        image_name + '_' + str(question_ids[idx]) \
                        + '_result.jpg'))

                idx += 1

        results = [{'question_id': question_id, 'answer': answer} \
                   for question_id, answer in zip(question_ids, answers)]
        fp = open(config.eval_result_file, 'wb')
        json.dump(results, fp)
        fp.close()

        # Evaluate these answers
        eval_res_vqa = eval_gt_vqa.loadRes(config.eval_result_file,
                                           config.eval_question_file)
        scorer = VQAEval(eval_gt_vqa, eval_res_vqa)
        scorer.evaluate()
        print("Evaluation complete.")

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images and questions. """
        print("Testing the model...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        question_ids = test_data.question_ids
        answers = []

        # Compute the answers to the questions
        idx = 0
        for k in tqdm(list(range(test_data.num_batches))):
            batch = test_data.next_batch()
            image_files, question_word_idxs, question_lens = batch
            feed_dict = self.get_feed_dict(batch)
            result = sess.run(self.prediction, feed_dict = feed_dict)

            fake_cnt = 0 if k < test_data.num_batches-1 \
                       else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                answer = vocabulary.words[result[l]]
                answers.append(answer)

                # Save the result in an image file
                image_file = image_files[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]

                q_word_idxs = question_word_idxs[l]
                q_len = question_lens[l]
                q_words = [vocabulary.words[q_word_idxs[i]] \
                    for i in range(q_len)]
                if q_words[-1] != '?':
                    q_words.append('?')
                Q = 'Q: ' + ''.join([' '+w if not w.startswith("'") \
                    and w not in string.punctuation \
                    else w for w in q_words]).strip()
                A = 'A: ' + answer

                image = mpimg.imread(image_file)
                plt.imshow(image)
                plt.axis('off')
                plt.title(Q+'\n'+A)
                plt.savefig(os.path.join(config.test_result_dir, \
                    image_name + '_' + str(question_ids[idx]) \
                    + '_result.jpg'))

                idx += 1

        # Save the answers to a file
        test_info = pd.read_csv(config.temp_test_info_file)
        results = pd.DataFrame({'question_id': question_ids,
                                'answer': answers})
        results = pd.merge(test_info, results)
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
