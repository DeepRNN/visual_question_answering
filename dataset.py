import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from utils.vocabulary import Vocabulary
from utils.vqa.vqa import VQA

class DataSet(object):
    def __init__(self,
                 image_files,
                 question_word_idxs,
                 question_lens,
                 question_ids,
                 batch_size,
                 answer_idxs = None,
                 is_train = False,
                 shuffle = False):
        self.image_files = np.array(image_files)
        self.question_word_idxs = np.array(question_word_idxs)
        self.question_lens = np.array(question_lens)
        self.question_ids = np.array(question_ids)
        self.answer_idxs = np.array(answer_idxs)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.question_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end]
            current_idxs += list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_files[current_idxs]
        question_word_idxs = self.question_word_idxs[current_idxs]
        question_lens = self.question_lens[current_idxs]

        if self.is_train:
            answer_idxs = self.answer_idxs[current_idxs]
            self.current_idx += self.batch_size
            return image_files, question_word_idxs, question_lens, answer_idxs
        else:
            self.current_idx += self.batch_size
            return image_files, question_word_idxs, question_lens

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count

def prepare_train_data(config):
    """ Prepare the data for training the model. """
    vqa = VQA(config.train_answer_file, config.train_question_file)
    vqa.filter_by_ques_len(config.max_question_length)
    vqa.filter_by_ans_len(1)

    print("Reading the questions and answers...")
    annotations = process_vqa(vqa,
                              'COCO_train2014',
                              config.train_image_dir,
                              config.temp_train_annotation_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    answers = annotations['answer'].values
    print("Questions and answers read.")
    print("Number of questions = %d" %(len(question_ids)))

    print("Building the vocabulary...")
    vocabulary = Vocabulary()
    if not os.path.exists(config.vocabulary_file):
        for question in tqdm(questions):
            vocabulary.add_words(word_tokenize(question))
        for answer in tqdm(answers):
            vocabulary.add_words(word_tokenize(answer))
        vocabulary.compute_frequency()
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))
    config.vocabulary_size = vocabulary.size

    print("Processing the questions and answers...")
    if not os.path.exists(config.temp_train_data_file):
        question_word_idxs, question_lens = process_questions(questions,
                                                              vocabulary,
                                                              config)
        answer_idxs = process_answers(answers, vocabulary)
        data = {'question_word_idxs': question_word_idxs,
                'question_lens': question_lens,
                'answer_idxs': answer_idxs}
        np.save(config.temp_train_data_file, data)
    else:
        data = np.load(config.temp_train_data_file).item()
        question_word_idxs = data['question_word_idxs']
        question_lens = data['question_lens']
        answer_idxs = data['answer_idxs']
    print("Questions and answers processed.")

    print("Building the dataset...")
    dataset = DataSet(image_files,
                      question_word_idxs,
                      question_lens,
                      question_ids,
                      config.batch_size,
                      answer_idxs,
                      True,
                      True)
    print("Dataset built.")
    return dataset, config

def prepare_eval_data(config):
    """ Prepare the data for evaluating the model. """
    vqa = VQA(config.eval_answer_file, config.eval_question_file)
    vqa.filter_by_ques_len(config.max_question_length)
    vqa.filter_by_ans_len(1)

    print("Reading the questions...")
    annotations = process_vqa(vqa,
                              'COCO_val2014',
                              config.eval_image_dir,
                              config.temp_eval_annotation_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    print("Questions read.")
    print("Number of questions = %d" %(len(question_ids)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))
    config.vocabulary_size = vocabulary.size

    print("Processing the questions...")
    if not os.path.exists(config.temp_eval_data_file):
        question_word_idxs, question_lens = process_questions(questions,
                                                              vocabulary,
                                                              config)
        data = {'question_word_idxs': question_word_idxs,
                'question_lens': question_lens}
        np.save(config.temp_eval_data_file, data)
    else:
        data = np.load(config.temp_eval_data_file).item()
        question_word_idxs = data['question_word_idxs']
        question_lens = data['question_lens']
    print("Questions processed.")

    print("Building the dataset...")
    dataset = DataSet(image_files,
                      question_word_idxs,
                      question_lens,
                      question_ids,
                      config.batch_size)
    print("Dataset built.")
    return vqa, dataset, vocabulary, config

def prepare_test_data(config):
    """ Prepare the data for testing the model. """
    print("Reading the questions...")
    annotations = pd.read_csv(config.test_question_file)
    images = annotations['image'].unique()
    image_files = [os.path.join(config.test_image_dir, f) for f in images]

    temp = pd.DataFrame({'image': images, 'image_file': image_files})
    annotations = pd.merge(annotations, temp)
    annotations.to_csv(config.temp_test_info_file)

    image_files = annotations['image_file'].values
    questions = annotations['question'].values
    question_ids = annotations['question_id'].values
    print("Questions read.")
    print("Number of questions = %d" %(len(question_ids)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))
    config.vocabulary_size = vocabulary.size

    print("Processing the questions...")
    question_word_idxs, question_lens = process_questions(questions,
                                                          vocabulary,
                                                          config)
    print("Questions processed.")

    print("Building the dataset...")
    dataset = DataSet(image_files,
                      question_word_idxs,
                      question_lens,
                      question_ids,
                      config.batch_size)
    print("Dataset built.")
    return dataset, vocabulary, config

def process_vqa(vqa, label, image_dir, annotation_file):
    """ Build a temporary annotation file for training or evaluation. """
    question_ids = list(vqa.qa.keys())
    image_ids = [vqa.qa[k]['image_id'] for k in question_ids]
    image_files = [os.path.join(image_dir, label+"_000000"+("%06d" %k)+".jpg")
                   for k in image_ids]
    questions = [vqa.qqa[k]['question'] for k in question_ids]
    answers = [vqa.qa[k]['best_answer'] for k in question_ids]

    annotations = pd.DataFrame({'question_id': question_ids,
                                'image_id': image_ids,
                                'image_file': image_files,
                                'question': questions,
                                'answer': answers})
    annotations.to_csv(annotation_file)
    return annotations

def process_questions(questions, vocabulary, config):
    """ Tokenize the questions and translate each token into its index \
        in the vocabulary, and get the number of tokens. """
    question_word_idxs = []
    question_lens = []
    for q in tqdm(questions):
        word_idxs = vocabulary.process_sentence(q)
        current_length = len(word_idxs)
        current_word_idxs = np.zeros((config.max_question_length), np.int32)
        current_word_idxs[:current_length] = np.array(word_idxs)
        question_word_idxs.append(current_word_idxs)
        question_lens.append(current_length)
    return np.array(question_word_idxs), np.array(question_lens)

def process_answers(answers, vocabulary):
    """ Translate the answers into their indicies in the vocabulary. """
    answer_idxs = []
    for answer in tqdm(answers):
        answer_idxs.append(vocabulary.word_to_idx(word_tokenize(answer)[0]))
    return np.array(answer_idxs)

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    vqa = VQA(config.train_answer_file, config.train_question_file)
    vqa.filter_by_ques_len(config.max_question_length)
    vqa.filter_by_ans_len(1)

    question_ids = list(vqa.qa.keys())
    questions = [vqa.qqa[k]['question'] for k in question_ids]
    answers = [vqa.qa[k]['best_answer'] for k in question_ids]

    vocabulary = Vocabulary()
    for question in tqdm(questions):
        vocabulary.add_words(word_tokenize(question))
    for answer in tqdm(answers):
        vocabulary.add_words(word_tokenize(answer))
    vocabulary.compute_frequency()
    vocabulary.save(config.vocabulary_file)
    return vocabulary
