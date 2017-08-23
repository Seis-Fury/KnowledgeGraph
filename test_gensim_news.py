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
生成lsi模型
'''

#stop_words = '/usr/local/NLP_pkg/jieba/extra_dict/stop_words.txt'
stop_words = './stop_words_ch.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf-8').readlines()
stopwords = [ w.strip() for w in stopwords ] 


stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
stop_dict = dict(zip(stopwords,stopwords))
stop_f = dict(zip(stop_flag,stop_flag))
#print stop_dict
#print stop_f
def Demo_time(func):
    def wrapper():
        b = time.time()
        func()
        e = time.time()
        secs = e - b
        print "->elpased time is %f " %secs
    return wrapper

def tokenization(filename):
    result = []
    with open(filename, 'r') as f:
        text = f.read()
        words = pseg.cut(text)
        for word, flag in words:
            if flag not in stop_f and word not in stop_dict:
                result.append(word)
        return result
def feed(data_list,num):
    with open('./BondDocs/'+str(num)+'.txt','w') as fp:
        for i in data_list:
            fp.write(i.encode('utf-8'))
            fp.write('\n')
def preprocess(path):

    #for i in xrange(1,200):
    corpus = []
    for i in os.listdir(path):
        if i.startswith('100') and i.endswith('txt'):
            inp = os.path.join(path, str(i))
            print inp
            text = tokenization(inp)
            corpus.append(text)
            print "-"*10+'No.'+str(i).strip('.txt')
            print text
        #feed(text,i)
    return corpus

def initmodel(corpus=None):
        dictionary = corpora.Dictionary(corpus)
        dictionary.save('./updown/new_dict.txt') #saved
        doc_vectors = [ dictionary.doc2bow(text) for text in corpus ] #生成词向量

        corpora.MmCorpus.serialize('./updown_new/new_corpus.mm',doc_vectors) #保存生成的词向量模型
        #corpora=corpora.MmCorpus('./updown/new_corpus.mm')
        tfidf_model = models.TfidfModel(doc_vectors)
        corpus_tfidf = tfidf_model[doc_vectors]

        lsi = models.LsiModel(corpus_tfidf,id2word=dictionary, num_topics=200)
        lsi.save('./updown/lsi_200.model')  
def load_model(path=None):
        #加载字典
        dictionary = corpora.Dictionary.load('./updown_new/new_dict.txt') #加载字典
        #加载语料
        corpus = corpora.MmCorpus('./updown_new/new_corpus.mm')
        #生成TF-IDF统计工作
        #tfidf = models.TfidfModel(corpus)
        #tfidf.save('./updown_new/tfidf.model')
        tfidf = models.TfidfModel.load('./updown_new/tfidf.model')
        #加载LSI模型
        lsi = models.LsiModel.load('./updown_new/lsi_200.model',mmap='r')
        return dictionary ,tfidf, lsi
def sent2lsi(sent, dictionary, tfidf, lsi):
    #分词
    sent_cut_raw = list(jieba.cut(sent))
    #转化成词袋模型
    #print '/'.join(sent_cut_raw)
    bow = dictionary.doc2bow(sent_cut_raw)
    #print bow
    #统计词频
    sent_tfidf = tfidf[bow]
    #求取lsi值
    lsi_mat = lsi[sent_tfidf]
    return lsi_mat
def TextCosine(sent1, sent2, dictionary, tfidf, lsi):
    #分句，并转换成lsi的词向量
    sent1_lsi = sent2lsi(sent1, dictionary, tfidf, lsi)
    sent2_lsi = sent2lsi(sent2, dictionary, tfidf, lsi)
    #print sent1_lsi

    Sum = 0.0
    sum_1 = 0.0
    sum_2 = 0.0
    #print len(sent1_lsi)
    if sent1_lsi == None or sent2_lsi == None:
        #print sent1_lsi
        #print sent2_lsi
        return 0.0
    elif len(sent1_lsi)!=200 or len(sent2_lsi)!=200:
        return 0.0
    else:
        for i in xrange(len(sent1_lsi)):
            Sum += sent1_lsi[i][1]*sent2_lsi[i][1]
            sum_1 += sent1_lsi[i][1] **2
            sum_2 += sent2_lsi[i][1] **2
        return Sum/(sqrt(sum_1*sum_2))

def login_mongo(db_name):
    """
    Mongo is loading...
    Be patient is A Good habit
    ---------------------------------------------
    """
    client = pymongo.MongoClient('localhost',27017)
    return client[db_name]

def test_init():
    path = '/Users/rich/Desktop/data/DATA'

    #text = codecs.open(fn,'r',encoding='utf-8').readlines()
    Corpus = preprocess(path)
    initmodel(Corpus)

def en_merge(Ftxt,entity1,entity2):
    t_s = time.time()
    name_s = ''
    id_1 = entity1['id']
    id_2 = entity2['id']
    if not entity1['c_id'] and not entity2['c_id']:
        fp = open('comon_id_index.txt','a+')
        name1 = entity1['name']
        name2 = entity2['name']
    
        if name1 in name2:
            name_s = name1.encode('utf-8') + str(t_s)
        else:
            name_s = name2.encode('utf-8') + str(t_s)

        common_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,name_s))+str(t_s)
         
        Ftxt.update({'id':id_1},{'$set':{'c_id':common_id}})
        Ftxt.update({'id':id_2},{'$set':{'c_id':common_id}})

        fp.write(str(common_id)+'\n')
    elif not entity1['c_id'] or not entity2['c_id']:
        if entity1['c_id']: Ftxt.update({'id':id_2},{'$set':{'c_id':entity1['c_id']}})
        else: Ftxt.update({'id':id_1},{'$set':{'c_id':entity2['c_id']}})
    elif entity1['c_id'] != entity2['c_id']:
        with open('default_Merge.txt','a+') as fp:
            fp.write(str(entity1['id'])+' and '+str(entity2['id'])+'\n')


def en_Similar(Ftxt, id_1 , id_2, d, t ,l ):
    sent1_d = Ftxt.find_one({'id':id_1})
    sent2_d = Ftxt.find_one({'id':id_2})
    sent1_li = sent1_d['sentenses']
    sent2_li = sent2_d['sentenses']
    #print id_1,id_2
    for sent1 in sent1_li:
        for sent2 in sent2_li:
            cc = TextCosine(sent1, sent2, d ,t, l)
            if cc > 0.8 :
                #print sent1
                #print sent2
                print 'Similar %f' %(cc)
                
                #merge
                en_merge(Ftxt, sent1_d, sent2_d)
                return cc
            else:
                pass
                #print 'Not same'

    return 0


def compare_name(name1,name2):
    if name1 == name2:
        return True
    elif name1 in name2:
        return True
    elif name2 in name1:
        return True
    return False

def build_cid(en_db):
    db = en_db
    try:
        val = db.find_one()["c_id"]
        while 1:
            y = raw_input('Do you want to rebuild: y<es> or n<es>')
            if y.strip().strip('\n').lower() == 'y':
                db.update({},{"$set":{'c_id':None}},multi=True)
                break
            elif y.strip().strip('\n').lower() == 'n':
                break

    except KeyError:
        db.update({},{"$set":{'c_id':None}},multi=True)
    
def check_cid(en_db):
    db = en_db
    func = '''
            function(obj, prev)
            {
                prev.count++;
            }
            '''
    with open('comon_id_index.txt','r') as fid:
        for line in fid:
            c_id = line.strip('\n').decode()
            print c_id,type(c_id)
            stat = db.group(key={"c_id":1},condition={"c_id":c_id},initial={"count":0},reduce=func)
            print stat


def main():
    #text = codecs.open(fn,'r',encoding='utf-8').readlines()
    
    #initmodel()
    #client = pymongo.MongoClient('localhost',27017)
    
 
    b = time.time()

    #Mongo Preprog

    db = login_mongo('bondmarket')
    en_db = db.copyentityFnews
    doc_db = db.copyparaFnews
    en_db.update({},{"$set":{'c_id':None}},multi=True)
    #build_cid(en_db)


    load_time = time.time()
    dt_model, tfidf_model, lsi_model = load_model()
    end_time = time.time()

    print 'model loading costs {0}s'.format(end_time-load_time)
    #Prepare
    cc_r = np.array([])
    
    for i in xrange(1,10):
        doc_i = doc_db.find({u'article_No':'0_'+str(i)})
        if not doc_i.count(): continue
        #print doc_i[0]
        #抽出里面的每一个段落
        for para_i in doc_i:

            id_i = para_i[u'entities_id']
            na_i = para_i[u'entities_name']
            for j in xrange(i+1,10):
                doc_j = doc_db.find({u'article_No':'0_'+str(j)})
                if not doc_j.count(): continue
                for para_j in doc_j:
                    id_j = para_j[u'entities_id']
                    na_j = para_j[u'entities_name']

                    for num1 in xrange(len(id_i)):
                        for num2 in xrange(len(id_j)):
                            if not compare_name(na_i[num1], na_j[num2]):
                                #print na_i[num1]
                                print para_i
                                print para_j
                                print id_i[num1],id_j[num2]
                                cc = en_Similar(en_db, id_i[num1], id_j[num2] ,dt_model,tfidf_model, lsi_model)
                                if cc:
                                #   print en_db.find_one({'id':id_i[num1]})
                                    #print en_db.find_one({'id':id_j[num2]})
                                    cc_r = np.append(cc_r,cc)
        
    print cc_r
    print len(cc_r)
    c = time.time()
    
    
    #print d-b
    #print c-d
    print "computation costs {0:.2f}s".format(c-b)
    
    res1,res2 = [],[]
    for x in cc_r:
        if x > 0.9:
            res1.append(x)
        else:
            res2.append(x)

    print len(res1),len(res2)
    return cc_r


def feed(c_id):
    '''Show the data which have save common id
    '''
    res = db.entityFtxt.find({"c_id":c_id})
    for i in res:
        print i['name'],i['id']
        for j in i['sentenses']:
            print j,
        print '\n'

if __name__ == '__main__':
    main()
    #db = login_mongo('bondmarket')
    #check_cid(db.entityFtxt)
    #load_model()
    #test_init()

