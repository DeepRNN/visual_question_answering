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
        self.dim_embed = dim_embed
        self.max_sent_len = max_sent_len
        self.save_file = save_file
        self.add_words(["<unknown>"])

    def load_glove(self, glove_dir):
        glove_file = os.path.join(glove_dir, 'glove.6B.'+str(self.dim_embed)+'d.txt')
        print("Loading Glove data from %s" %(glove_file))

        with open(glove_file) as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = [float(x) for x in l[1:]]

        print("Glove data loaded")

    def add_words(self, words):
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = self.num_words
                self.num_words += 1
            if w not in self.word2vec:
                self.word2vec[w] = np.random.uniform(0.0, 1.0, (self.dim_embed))

    def filter_word2vec(self):
        word2vec = {}
        for w in self.word2idx:
            word2vec[w] = self.word2vec[w] 
        self.word2vec = word2vec

    def word_to_index(self, word):
        return self.word2idx[word] if word in self.word2idx else 0
       
    def symbolize_sent(self, sent):
        indices = np.zeros(self.max_sent_len).astype(np.int32)
        words = np.array([self.word_to_index(w) for w in sent.lower().split(' ')])
        indices[:len(words)] = words
        return indices, len(words)

    def save(self):
        pickle.dump([self.idx2word, self.word2idx, self.word2vec, self.num_words], open(self.save_file, 'wb'))

    def load(self):
        self.idx2word, self.word2idx, self.word2vec, self.num_words = pickle.load(open(self.save_file, 'rb'))

