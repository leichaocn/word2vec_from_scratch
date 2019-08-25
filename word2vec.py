# -*- coding: utf-8 -*-
#!/usr/bin/env python
# File : word2vec.py
# Date : 2019/8/25
# Author: leichao
# Email : leichaocn@163.com

"""简述功能.

详细描述.
"""

__filename__ = "word2vec.py"
__date__ = 2019 / 8 / 25
__author__ = "leichao"
__email__ = "leichaocn@163.com"

import os
import sys

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple word2vec from scratch with Python
#   2018-FEB
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
import re
from collections import defaultdict


# --- CONSTANTS ----------------------------------------------------------------+


class word2vec():
    def __init__(self,settings):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # FORWARD PASS
    # 前向传播
    def forward_pass(self, x):
        """前向传播
        x是输入，h是隐层加权和，u是输出层的加权和，y_c是输出层的激活值
        注意：隐层没有激活函数。
        """
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    # 反向传播
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        # 初始化隐层的权重矩阵
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))  # embedding matrix
        # 初始化输出层的权重矩阵
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # context matrix

        # self.loss = 0
        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                # self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))

            # 每一轮epoch的训练，都会将权重调整向损失减少的方向。
            # 因此越往后的epoch，LOSS越小。一
            print('EPOCH:', i, 'LOSS:', self.loss)
        pass

    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        # words_sorted = sorted(word_sim.items(), key=lambda (word, sim): sim, reverse=True)
        words_sorted = sorted(word_sim.items(), key=lambda word_sim: word_sim[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print (word, sim)

        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        # words_sorted = sorted(word_sim.items(), key=lambda (word, sim): sim, reverse=True)
        words_sorted = sorted(word_sim.items(), key=lambda word_sim: word_sim[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass


def main():
    """仅用于单元测试"""
    # 配置初始信息
    settings = {}
    settings['n'] = 5  # 隐层的维度
    settings['window_size'] = 2  # skip-gram的窗口大小，为2，表示最多前后哥
    settings['min_count'] = 0  # minimum word count
    settings['epochs'] = 5000  # number of training epochs
    # settings['neg_samp'] = 10
    settings['learning_rate'] = 0.01  # learning rate
    np.random.seed(0)  # set the seed for reproducibility
    # 由一句话组成的语料
    corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

    # I
    w2v = word2vec(settings)

    """由语料来生成训练数据
    因为原始语料有9个词，因此生成的训练数据对 也是9条
    每条训练数据，由2个数组构成;
    第一个数组，是输入的词，所以是一个one-hot向量
    第二个数组，是label，由skip-gram能涵盖的若干个词构成，所以是若干个one-hot向量。
    """
    training_data = w2v.generate_training_data(corpus)


    # 训练
    w2v.train(training_data)

if __name__ == '__main__':
    main()