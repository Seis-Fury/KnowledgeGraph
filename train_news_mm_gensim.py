#/usr/bin/env python
#-*- coding:utf-8 -*-
import jieba
import codecs
import os
import time
import pymongo
import numpy as np
from jieba import posseg as pseg
from gensim import corpora, similarities, models
from math import *
import uuid

'''
用新闻文本训练Lsi和Lda模型
'''

stop_words = './stop_words_ch.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf-8').readlines()
stopwords = [ w.strip() for w in stopwords ] 


stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
stop_dict = dict(zip(stopwords,stopwords))
stop_f = dict(zip(stop_flag,stop_flag))

def Demo_time(func):
    def wrapper():
        b = time.time()
        func()
        e = time.time()
        secs = e - b
        print "->elpased time is %f " %secs
    return wrapper

def tokenization(filename):
    with open(filename, 'r') as f:
        text = f.read()
        for line in text.split('\n'):
            result = []
            words = pseg.cut(line)
            for word, flag in words:
                if flag not in stop_f and word not in stop_dict:
                    result.append(word)
            yield result
def preprocess(path):
    corpus = []
    for u in xrange(0,4):
        path_u = path+'/'+str(u)+'x' 
        for i in os.listdir(path_u):
            if i.endswith('_1.txt'):
                inp = os.path.join(path_u, str(i))
                print inp
                for text in  tokenization(inp):
                    corpus.append(text)
                    print text
                print "-"*10+'No.'+str(i).strip('.txt') 
    return corpus
def initmodel_lsi(corpus=None):
        dictionary = corpora.Dictionary(corpus)
        dictionary.save('./updown/news_mm_dict.txt') #saved
        doc_vectors = [ dictionary.doc2bow(text) for text in corpus ] #生成词向量

        corpora.MmCorpus.serialize('./updown_new/news_mm_corpus.mm',doc_vectors) #保存生成的词向量模型
        #corpora=corpora.MmCorpus('./updown/new_corpus.mm')
        tfidf_model = models.TfidfModel(doc_vectors)
        tfidf_model.save('./updown_new/news_tfidf.mm')
        corpus_tfidf = tfidf_model[doc_vectors]

        lsi = models.LsiModel(corpus_tfidf,id2word=dictionary, num_topics=200)
        lsi.save('./updown/news_lsi_200.model')  
def init_model_lda(corpus=None):
    dictionary = corpora.Dictionary.load('./updown/news_mm_dict.txt')
    print 'dict load Successfullt'
    corpus = corpora.MmCorpus('./updown/news_mm_corpus.mm')
    print 'corpus load successfully'
    tfidf_model = models.load('./updown_new/news_tfidf.mm')
    corpus_tfidf = tfidf_model[corpus]

    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    lda.save('./updown_new/news_lda_100.mm')

@Demo_time
def main_init():
    path = '.'

    #text = codecs.open(fn,'r',encoding='utf-8').readlines()
    Corpus = preprocess(path)
    initmodel(Corpus)

    init_model_lda()
if __name__ == '__main__':
    main_init()
