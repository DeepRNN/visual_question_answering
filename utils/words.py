import os
import json
import numpy as np
import cPickle as pickle

class WordTable():
    def __init__(self, dim_embed, max_sent_len, save_file):
        self.idx2word = []
        self.word2idx = {}
        self.word2vec = {}
        self.num_words = 0
        self.word_count = {}
        self.word_freq = []
        self.dim_embed = dim_embed
        self.max_sent_len = max_sent_len
        self.save_file = save_file
        self.add_words(["<unknown>"])

    def add_words(self, words):
        """ Add new words to the vocabulary. """
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = self.num_words
                self.num_words += 1
            if w not in self.word2vec:
                self.word2vec[w] = np.random.randn(self.dim_embed) * 0.01
            self.word_count[w] = self.word_count.get(w, 0) + 1

    def compute_freq(self):
        """ Compute the frequency of each word. """
        self.word_freq = []
        for w in self.word2idx:
            self.word_freq.append(self.word_count[w])
        self.word_freq = np.array(self.word_freq, np.float32)
        self.word_freq /= np.sum(self.word_freq)
        self.word_freq = np.log(self.word_freq)
        self.word_freq -= np.max(self.word_freq)

    def filter_word2vec(self):
        """ Remove unseen words from the word embedding. """
        word2vec = {}
        for w in self.word2idx:
            word2vec[w] = self.word2vec[w] 
        self.word2vec = word2vec

    def word_to_index(self, word):
        """ Translate a word into its index. """
        return self.word2idx[word] if word in self.word2idx else 0
       
    def symbolize_sent(self, sent):
        """ Translate a sentence into the indicies of its words. """
        indices = np.zeros(self.max_sent_len).astype(np.int32)
        words = np.array([self.word_to_index(w) for w in sent.lower().split(' ')])
        indices[:len(words)] = words
        return indices, len(words)

    def save(self):
        """ Save the word table to pickle. """
        pickle.dump([self.idx2word, self.word2idx, self.word2vec, self.num_words, self.word_freq], open(self.save_file, 'wb'))

    def load(self):
        """ Load the word table from pickle. """
        self.idx2word, self.word2idx, self.word2vec, self.num_words, self.word_freq = pickle.load(open(self.save_file, 'rb'))

    def load_glove(self, glove_dir):
        """ Initialize the word embedding with GloVe data. """
        glove_file = os.path.join(glove_dir, 'glove.6B.'+str(self.dim_embed)+'d.txt')
        print("Loading GloVe data from %s" %(glove_file))
        with open(glove_file) as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = [float(x)*0.05 for x in l[1:]]
        print("GloVe data loaded")


