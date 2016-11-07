import os
import math
import numpy as np
import pandas as pd
import cPickle as pickle
import skimage
import skimage.io

from utils.words import *
from utils.vqa.vqa import *

class DataSet():
    def __init__(self, img_files, questions, question_lens, question_ids, answers=None, batch_size=1, is_train=False,  shuffle=False):
        self.img_files = np.array(img_files)
        self.questions = np.array(questions)
        self.question_lens = np.array(question_lens)
        self.question_ids = np.array(question_ids)
        self.answers = np.array(answers)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.question_ids)
        self.num_batches = int(self.count * 1.0 / self.batch_size)
        self.current_index = 0
        self.indices = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_idx = self.indices[start:end]

        img_files = self.img_files[current_idx]
        questions = self.questions[current_idx]
        question_lens = self.question_lens[current_idx]
        if self.is_train: 
            answers = self.answers[current_idx]
            self.current_index += self.batch_size
            return img_files, questions, question_lens, answers
        else:
            self.current_index += self.batch_size
            return img_files, questions, question_lens

    def has_next_batch(self):
        """ Determine whether there is any batch left. """
        return self.current_index + self.batch_size <= self.count


def prepare_train_data(args):
    """ Prepare relevant data for training the model. """
    image_dir, question_file, answer_file, annotation_file = args.train_image_dir, args.train_question_file, args.train_answer_file, args.train_annotation_file

    word_table_file, init_embed_with_glove, glove_dir = args.word_table_file, args.init_embed_with_glove, args.glove_dir
    dim_embed, batch_size, max_ques_len = args.dim_embed, args.batch_size, args.max_ques_len

    vqa = VQA(answer_file, question_file)
    vqa.filter_by_ques_len(max_ques_len)
    vqa.filter_by_ans_len(1)

    annotations = process_vqa(vqa, 'COCO_train2014', image_dir, annotation_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    answers = annotations['answer'].values
    print("Number of training questions = %d" %(len(question_ids)))

    print("Building the word table...")
    word_table = WordTable(dim_embed, max_ques_len, word_table_file)
    if not os.path.exists(word_table_file):
        if init_embed_with_glove:
            word_table.load_glove(glove_dir)
        for ques in questions:
            word_table.add_words(ques.split(' '))
        for ans in answers:
            word_table.add_words(ans.split(' '))
        word_table.filter_word2vec()
        word_table.compute_freq()
        word_table.save()
    else:
        word_table.load()
    print("Word table built. Number of words = %d." %(word_table.num_words))

    questions, question_lens = symbolize_questions(questions, word_table)
    answers = symbolize_answers(answers, word_table)

    print("Building the training dataset...")
    dataset = DataSet(image_files, questions, question_lens, question_ids, answers, batch_size, True, True)
    print("Dataset built.")
    return vqa, dataset

def prepare_val_data(args):
    """ Prepare relevant data for validating the model. """
    image_dir, question_file, answer_file, annotation_file = args.val_image_dir, args.val_question_file, args.val_answer_file, args.val_annotation_file

    word_table_file, glove_dir = args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_ques_len = args.dim_embed, args.batch_size, args.max_ques_len

    vqa = VQA(answer_file, question_file)
    vqa.filter_by_ques_len(max_ques_len)
    vqa.filter_by_ans_len(1)

    annotations = process_vqa(vqa, 'COCO_val2014', image_dir, annotation_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    print("Number of validation questions = %d" %(len(question_ids)))

    word_table = WordTable(dim_embed, max_ques_len, word_table_file)
    word_table.load()

    questions, question_lens = symbolize_questions(questions, word_table)
   
    print("Building the validation dataset...")
    dataset = DataSet(image_files, questions, question_lens, question_ids)
    print("Dataset built.")
    return vqa, dataset


def prepare_test_data(args):
    """ Prepare relevant data for testing the model. """
    image_dir, question_file = args.test_image_dir, args.test_question_file
    info_file = args.test_info_file

    word_table_file, glove_dir = args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_ques_len = args.dim_embed, args.batch_size, args.max_ques_len

    annotations = pd.read_csv(question_file)

    images = annotations['image'].unique()
    image_files = [os.path.join(image_dir, f) for f in images]
    
    temp = pd.DataFrame({'image': images, 'image_file': image_files})
    annotations = pd.merge(annotations, temp)
    annotations.to_csv(info_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    print("Number of testing questions = %d" %(len(question_ids)))

    word_table = WordTable(dim_embed, max_ques_len, word_table_file)
    word_table.load()

    questions, question_lens = symbolize_questions(questions, word_table)

    print("Building the testing dataset...")    
    dataset = DataSet(image_files, questions, question_lens, question_ids)
    print("Dataset built.")
    return dataset


def process_vqa(vqa, label, img_dir, annotation_file):
    """ Build an annotation file containing the training or validation information. """
    question_ids = list(vqa.qa.keys())
    image_ids = [vqa.qa[k]['image_id'] for k in question_ids]
    image_files = [os.path.join(img_dir, label+"_000000"+("%06d" %k)+".jpg") for k in image_ids]
    answers = [vqa.qa[k]['best_answer'] for k in question_ids]
    questions = [vqa.qqa[k]['question'] for k in question_ids]

    annotations = pd.DataFrame({'question_id': question_ids, 'image_id': image_ids, 'image_file': image_files, 'question': questions, 'answer': answers})
    annotations.to_csv(annotation_file)
    return annotations


def symbolize_questions(questions, word_table):
    """ Translate the questions into the indicies of their words in the vocabulary, and get their lengths. """
    ques_idxs = []
    ques_lens = []
    for q in questions:
        q_idx, q_len = word_table.symbolize_sent(q)
        ques_idxs.append(q_idx)
        ques_lens.append(q_len)
    return np.array(ques_idxs), np.array(ques_lens)


def symbolize_answers(answers, word_table):
    """ Translate the answers into their indicies in the vocabulary. """
    ans_indices = [word_table.word_to_index(ans.split(' ')[0]) for ans in answers]
    return np.array(ans_indices)

