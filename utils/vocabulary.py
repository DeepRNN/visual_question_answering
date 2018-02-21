import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

class Vocabulary(object):
    def __init__(self, save_file = None):
        self.words = []
        self.word2idx = {}
        self.size = 0
        self.word_counts = {}
        self.word_frequencies = []
        if save_file is not None:
            self.load(save_file)
        else:
            self.add_words(["<unknown>"])

    def add_words(self, words):
        """ Add new words to the vocabulary. """
        for w in words:
            if w not in self.word2idx.keys():
                self.words.append(w)
                self.word2idx[w] = self.size
                self.size += 1
            self.word_counts[w] = self.word_counts.get(w, 0) + 1

    def compute_frequency(self):
        """ Compute the frequency of each word. """
        self.word_frequencies = []
        for w in self.words:
            self.word_frequencies.append(self.word_counts[w])
        self.word_frequencies = np.array(self.word_frequencies, np.float32)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def word_to_idx(self, word):
        """ Translate a word into its index. """
        return self.word2idx[word] if word in self.word2idx.keys() else 0

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = word_tokenize(sentence.lower())
        word_idxs = [self.word_to_idx(w) for w in words]
        return word_idxs

    def save(self, save_file):
        """ Save the vocabulary to a file. """
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.size)),
                             'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.size = len(self.words)
        self.word2idx = {self.words[i]:i for i in range(self.size)}
        self.word_frequencies = data['frequency'].values
