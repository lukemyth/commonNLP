#!/usr/bin/python3

#@File: training2-2.py

#-*-coding:utf-8-*-

#@Author:MaYt_pc

#@Time: 2020年11月12日15时

#说明:
import datetime
import pickle
from numba import jit
from collections import Counter
import numpy as np
import joblib
import keras.backend as K
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Bidirectional, Input, BatchNormalization, Activation, Dropout, Dense, CuDNNGRU, \
    CuDNNLSTM, Conv1D, MaxPool1D, concatenate, Flatten
from keras.optimizers import Adam
import os
import random
import re
import sys
import jieba
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#选择GPU编号，如果只有一个GPU，为0
from sklearn.metrics import classification_report
import pandas as pd
title_len = 50#标题长度，如果只有一个输入，则不用管
content_plen = 2#文本分块个数，content_plen*100是文本长度
content_len = 100*content_plen#文本总长度
word_num = 8000#词表中字的个数

class Config():
    def __init__(self):
        self.train_file = 'data/train.xlsx'#训练数据文件，xlsx或csv
        self.eval_file = 'data/test.xlsx'#验证数据文件，xlsx或csv
        self.test_file = 'data/test.xlsx'#训练数据文件，xlsx或csv
        self.do_train = False#True#如果进行训练就为True
        self.do_eval = True#如果要验证数据就为True
        self.do_test = True# 如果要预测就为True
        self.title_name = 'news_title'#标题字段名，和数据文件字段名一致，如果没有填NoneNone
        self.content_name = 'news_content'#正文字段名，和数据文件字段名一致，只有一个输入则是文本字段名
        self.label1_name = '情感大类'#输出1字段名，和数据文件字段名一致
        self.label2_name = None#'情感小类'#输出2字段名，和数据文件字段名一致，如果没有一定填None
        self.label1_dicfile = 'dl_id.txt'#标签1字典，必填
        self.label2_dicfile = 'xl_id.txt'#标签2字典，如果没有填None
        self.lr = 0.0001#学习率
        self.epochs = 4
        self.batch_size = 128
        self.verbose = 5
        self.label1_id = dict()
        self.label2_id = dict()
        self.id_label1 = dict()
        self.tokenizer_file = 'tokenizer.model'
        self.id_label2 = dict()
        self.output = 'output_combine'
        trained_model_dir = None#如果需要在之前训练的模型上微调，则写模型的路径
        assert self.do_train|self.do_eval|self.do_test,'do_train,do_eval和do_test中至少有一个要为True'
        assert self.label1_dicfile,'config.label1_dicfile不能为空'
        #初始化
        self.tokenizer = joblib.load(self.tokenizer_file)
        if  os.path.exists(self.label1_dicfile):#如果标签字典文件存在，就直接读取
            with open(self.label1_dicfile,'r',encoding='utf-8')as f:
                self.label1_id = eval(f.read())
                self.id_label1 = {self.label1_id[x]:x for x in self.label1_id}
                if self.label2_name:
                    with open(self.label2_dicfile,'r',encoding='utf-8')as f:
                        self.label2_id = eval(f.read())
                        self.id_label2 = {self.label2_id[x]:x for x in self.label2_id}
        else:#如果标签字典文件不存在，就根据数据得出并写入文件
            assert os.path.exists(self.train_file) | os.path.exists(self.eval_file),f'self.label1_dicfile不存在，且训练集和验证集不存在'
            if os.path.exists(self.train_file):#如果存在训练集，读取训练集数据
                data = read_data(self.train_file)
                if os.path.exists(self.eval_file):#如果存在验证集，将验证集和训练集拼接
                    data = pd.concat([data,read_data(self.eval_file)])
            else:
                data = read_data(self.eval_file)#如果存在验证集，读取验证集数据
            print(85,data.columns)
            labels1 = data[self.label1_name].unique()#取出标签1
            self.label1_id = {i:j for i,j in zip(labels1,range(len(labels1)))}
            self.id_label1 = {self.label1_id[x]:x for x in self.label1_id}
            with open(self.label1_dicfile,'w',encoding='utf-8') as f:#写入标签1字典文件
                f.write(str(self.label1_id))
            if self.label2_name:
                labels2 = data[self.label2_name].unique()#取出标签2
                self.label2_id = {i:j for i,j in zip(labels2,range(len(labels2)))}
                self.id_label2 = {self.label2_id[x]:x for x in self.label2_id}
                with open(self.label2_dicfile,'w',encoding='utf-8') as f:#写入标签2字典文件
                    f.write(str(self.label2_id)) 
                    
def read_data(data_file):
    if data_file[-3:] == 'csv':
        data = pd.read_csv(data_file)
    else:
        data = pd.read_excel(data_file)
    return data
    
def clean_text(text):#文本清洗
    text = str(text)
    text = re.sub(r'https{0,1}://[^ ]+', "", text)
    text = re.sub(r'/{0,1}/{0,1}@[^:： ]{1,20}(:|：| )', " ", text) ##去除微博昵称
    text = str(text)
    restr = '[\s+\/,%^*\-+]+|[+——~@#%……&*]+'
    resu = text.replace('&nbsp;', '').replace('ldquo', '“').replace('rdquo', '”') \
        .replace('lsquo', '‘').replace('rsquo', '’').replace('〔', '（').replace('〕', '）').replace('/', '') \
        .replace('&middot;', '·').replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    resu = re.split(r'\s+', resu)
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', ''.join(resu))
    line = re.sub(restr, '', dd)
    eng = [",", "!", "?", ":", ";", "(", ")", "[", "]", "$", "。。"]
    chi = ["，", "！", "？", "：", "；", "（", "）", "【", "】", "￥", '。']
    for i, j in zip(eng, chi):
        line = line.replace(i, j)
    lens = int(len(text)/content_plen)
    if lens > 100:
        text = ""
        for i in range(content_plen):
            text += line[i*lens:(1+i)*lens][:100]
#             text += line[i*lens:(1+i)*lens][::-1][:100]
        line = text
    return list(line)


#数据预处理
def preprocess_train(config,data_file):    
    data = read_data(data_file)
    print(134,data.columns)
    #进行正文文本处理
    content = data[config.content_name]   
    content = [clean_text(f) for f in content]
    content = config.tokenizer.texts_to_sequences(content)
    content = pad_sequences(content, maxlen=content_len, padding='post', truncating='post')    
    if config.title_name:
        #进行标题文本处理
        title = data[config.title_name]   
        title = [clean_text(f) for f in title]
        title = config.tokenizer.texts_to_sequences(title)
        title = pad_sequences(title, maxlen=title_len, padding='post', truncating='post')    
    #进行标签1处理
    y1 = data[config.label1_name]#从数据中取出标签1
    y1 = [config.label1_id[i] for i in y1]#将标签转化为编号
    label1 = np.zeros((len(content),len(config.label1_id)))
    for i in range(len(y1)):
        label1[i,y1[i]] = 1
    if config.label2_name:#如果标签2存在
        y2 = data[config.label2_name]#从数据中取出标签2
        y2 = [config.label2_id[i] for i in y2]#将标签2转化为编号
        label2 = np.zeros((len(content),len(config.label2_id)))
        for i in range(len(y2)):
            label2[i,y2[i]] = 1
        if config.title_name:
            return title,content,label1,label2
        else:
            return content,label1,label2
    else:
        if config.title_name:
            return title,content,label1
        return content,label1
def preprocess_test(config,data_file):    
    if data_file[-3:] == 'csv':
        data = pd.read_csv(data_file)
    else:
        data = pd.read_excel(data_file)
    #进行正文文本处理
    content = data[config.content_name]   
    content = [clean_text(f) for f in content]
    content = config.tokenizer.texts_to_sequences(content)
    content = pad_sequences(content, maxlen=content_len, padding='post', truncating='post')    
    if config.title_name:
        #进行标题文本处理
        title = data[config.title_name]   
        title = [clean_text(f) for f in title]
        title = config.tokenizer.texts_to_sequences(title)
        title = pad_sequences(title, maxlen=title_len, padding='post', truncating='post')    
    if config.title_name:
        return title,content
    else:
        return content

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
# from elmoformanylangs import Embedder
# e = Embedder('./zhs.model/')
def do_measures(id_label,true_matrix,pred_matrix):
    true = [id_label[list(x).index(max(x))] for x in true_matrix]
    pred = [id_label[list(x).index(max(x))] for x in pred_matrix]
    print(classification_report(true,pred))
    return true,pred
def create_model(config):
    #正文的网络结构
    content_input = Input(shape= (content_len,),name = 'content')    
    emb_c = Embedding(config.tokenizer.num_words+1, 512)(content_input)   
    pool_output = []
    kernel_sizes = [2, 3, 4, 5]
    for kernel_size in kernel_sizes:
#         emb_c = Dropout(0.3)(emb_c)
        c = Conv1D(filters=64, kernel_size=kernel_size, strides=1)(emb_c)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
    pool_output = concatenate([p for p in pool_output])
    normal = BatchNormalization()(pool_output)
    act = Activation("relu")(normal)
    feature_content = Flatten()(act)
    if not config.title_name:
        feature = feature_content
    else:
        title_input = Input(shape= (title_len,),name = 'title')   
        emb_t = Embedding(config.tokenizer.num_words+1, 512)(title_input)
        pool_output = []
        kernel_sizes = [ 2, 3, 4, 5]
        for kernel_size in kernel_sizes:
    #         emb_t = Dropout(0.3)(emb_t)
            c = Conv1D(filters=64, kernel_size=kernel_size, strides=1)(emb_t)
            p = MaxPool1D(pool_size=int(c.shape[1]))(c)
            pool_output.append(p)
        pool_output = concatenate([p for p in pool_output])
        normal = BatchNormalization()(pool_output)
        act = Activation("relu")(normal)
        feature_title = Flatten()(act)    
        feature = concatenate([feature_title,feature_content])    
    output1 = Dense(len(config.label1_id), activation='softmax',name="dl")(feature)
    if config.label2_name:
        output2 = Dense(len(config.label2_id), activation='softmax',name="xl")(feature)
        if config.title_name:
            model = Model(inputs = [title_input,content_input], outputs=[output1,output2])
        else:
            model = Model(inputs = [content_input], outputs=[output1,output2])
    else:
        if config.title_name:
            model = Model(inputs = [title_input,content_input], outputs=[output1])
        else:
            model = Model(inputs = [content_input], outputs=[output1])            
    return model
if __name__ == '__main__':
    config = Config() 
    if not os.path.exists(config.output):
        os.makedirs(config.output)#如果没有输出文件夹则新建
    #构造模型
    model = create_model(config)
    model.summary()
    if config.label2_name:
        save_best = ModelCheckpoint(f'{config.output}/model.h5', verbose=1, monitor='val_dl_fmeasure',mode = "max",save_best_only=True, save_weights_only=True)
    else:
        save_best = ModelCheckpoint(f'{config.output}/model.h5', verbose=1, monitor='val_fmeasure',mode = "max",save_best_only=True, save_weights_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(config.lr), metrics=[precision,fmeasure])
    if config.trained_model_dir:
        model.load_weights(config.trained_model_dir)
    if config.do_train:        
        if config.title_name:
            if config.label2_name:            
                train_title,train_content,train_y1,train_y2 = preprocess_train(config,config.train_file)        
                if config.do_eval:#如果有验证集，则训练完验证
                    eval_title,eval_content,eval_y1,eval_y2 = preprocess_train(config,config.eval_file)
                    model.fit([train_title,train_content], [train_y1,train_y2],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs,validation_data=([eval_title,eval_content],[eval_y1,eval_y2]), callbacks=[save_best])
                    model.load_weights(f"{config.output}/model.h5")
                    pres = model.predict([eval_title,eval_content], verbose=0)                    
                    true1,pred1 = do_measures(config.id_label1,eval_y1,pres[0])
                    true2,pred2 = do_measures(config.id_label2,eval_y2,pres[1])
                    eval_data = read_data(config.eval_file)
                    eval_data['pre1'] = pred1
                    eval_data['pre2'] = pred2
                    eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
                else:#如果没有验证集，训练完就结束                    
                    model.fit([train_title,train_content], [train_y1,train_y2],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs, callbacks=[save_best])                    
            else:#如果没有标签2
                train_title,train_content,train_y1 = preprocess_train(config,config.train_file)        
                if config.do_eval:#如果有验证集，则训练完验证
                    eval_title,eval_content,eval_y1 = preprocess_train(config,config.eval_file)
                    model.fit([train_title,train_content], [train_y1],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs,validation_data=([eval_title,eval_content],[eval_y1]), callbacks=[save_best])
                    model.load_weights(f"{config.output}/model.h5")
                    pres = model.predict([eval_title,eval_content], verbose=0)                    
                    true1,pred1 = do_measures(config.id_label1,eval_y1,pres)
                    eval_data = read_data(config.eval_file)
                    eval_data['pre1'] = pred1
                    eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
                else:#如果没有验证集，训练完就结束                    
                    model.fit([train_title,train_content], [train_y1],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs, callbacks=[save_best])
        else:#如果只有一个文本，没有标题
            if config.label2_name:            
                train_content,train_y1,train_y2 = preprocess_train(config,config.train_file)        
                if config.do_eval:#如果有验证集，则训练完验证
                    eval_content,eval_y1,eval_y2 = preprocess_train(config,config.eval_file)
                    model.fit([train_content], [train_y1,train_y2],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs,validation_data=([eval_content],[eval_y1,eval_y2]), callbacks=[save_best])
                    model.load_weights(f"{config.output}/model.h5")
                    pres = model.predict([eval_content], verbose=0)                    
                    true1,pred1 = do_measures(config.id_label1,eval_y1,pres[0])
                    true2,pred2 = do_measures(config.id_label2,eval_y2,pres[1])
                    eval_data = read_data(config.eval_file)
                    eval_data['pre1'] = pred1
                    eval_data['pre2'] = pred2
                    eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
                else:#如果没有验证集，训练完就结束                    
                    model.fit([train_content], [train_y1,train_y2],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs, callbacks=[save_best])
            else:#如果没有标签2
                train_content,train_y1 = preprocess_train(config,config.train_file)        
                if config.do_eval:#如果有验证集，则训练完验证
                    eval_content,eval_y1 = preprocess_train(config,config.eval_file)
                    model.fit([train_content], [train_y1],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs,validation_data=([eval_content],[eval_y1]), callbacks=[save_best])
                    model.load_weights(f"{config.output}/model.h5")
                    pres = model.predict([eval_content], verbose=0)                    
                    true1,pred1 = do_measures(config.id_label1,eval_y1,pres)   
                    eval_data = read_data(config.eval_file)
                    eval_data['pre1'] = pred1
                    eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
                else:#如果没有验证集，训练完就结束                    
                    model.fit([train_content], [train_y1],verbose = config.verbose,batch_size=config.batch_size, epochs=config.epochs, callbacks=[save_best])
    elif config.do_eval:#如果有测试集        
        if config.title_name:#如果有题目
            if config.label2_name: #如果标签2           
                eval_title,eval_content,eval_y1,eval_y2 = preprocess_train(config,config.eval_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([eval_title,eval_content], verbose=0)                    
                true1,pred1 = do_measures(config.id_label1,eval_y1,pres[0])
                true2,pred2 = do_measures(config.id_label2,eval_y2,pres[1])
                eval_data = read_data(config.eval_file)
                eval_data['pre1'] = pred1
                eval_data['pre2'] = pred2
                eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
            else:#如果没有标签2
                eval_title,eval_content,eval_y1 = preprocess_train(config,config.eval_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([eval_title,eval_content], verbose=0)                    
                true1,pred1 = do_measures(config.id_label1,eval_y1,pres)
                eval_data = read_data(config.eval_file)
                eval_data['pre1'] = pred1
                eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
        else:#如果没有题目
            if config.label2_name: #如果标签2           
                eval_content,eval_y1,eval_y2 = preprocess_train(config,config.eval_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([eval_content], verbose=0)                    
                true1,pred1 = do_measures(config.id_label1,eval_y1,pres[0])
                true2,pred2 = do_measures(config.id_label2,eval_y2,pres[1])
                eval_data = read_data(config.eval_file)
                eval_data['pre1'] = pred1
                eval_data['pre2'] = pred2
                eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
            else:#如果没有标签2
                eval_content,eval_y1 = preprocess_train(config,config.eval_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([eval_content], verbose=0)                    
                true1,pred1 = do_measures(config.id_label1,eval_y1,pres)
                eval_data = read_data(config.eval_file)
                eval_data['pre1'] = pred1
                eval_data.to_excel(f'{config.output}/output_eval.xlsx',index=False)
    if config.do_test:#如果有测试集
        if config.title_name:#如果有题目
            if config.label2_name: #如果标签2           
                test_title,test_content = preprocess_test(config,config.test_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([test_title,test_content], verbose=0)                    
#                 true1,pred1 = do_measures(config.id_label1,test_y1,pres[0])
#                 true2,pred2 = do_measures(config.id_label2,test_y2,pres[1])
                test_data = read_data(config.test_file)
                test_data['pre1'] = pred1
                test_data['pre2'] = pred2
                test_data.to_excel(f'{config.output}/output_test.xlsx',index=False)
            else:#如果没有标签2
                test_title,test_content = preprocess_test(config,config.test_file)                        
                model.load_weights(f"{config.output}/model.h5")
#                 pres = model.predict([test_title,test_content], verbose=0)                    
#                 true1,pred1 = do_measures(config.id_label1,test_y1,pres)
                test_data = read_data(config.test_file)
                test_data['pre1'] = pred1
                test_data.to_excel(f'{config.output}/output_test.xlsx',index=False)
        else:#如果没有题目
            if config.label2_name: #如果标签2           
                test_content = preprocess_test(config,config.test_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([test_content], verbose=0)                    
#                 true1,pred1 = do_measures(config.id_label1,test_y1,pres[0])
#                 true2,pred2 = do_measures(config.id_label2,test_y2,pres[1])
                test_data = read_data(config.test_file)
                test_data['pre1'] = pred1
                test_data['pre2'] = pred2
                test_data.to_excel(f'{config.output}/output_test.xlsx',index=False)
            else:#如果没有标签2
                test_content = preprocess_test(config,config.test_file)                        
                model.load_weights(f"{config.output}/model.h5")
                pres = model.predict([test_content], verbose=0)                    
#                 true1,pred1 = do_measures(config.id_label1,test_y1,pres)
                test_data = read_data(config.test_file)
                test_data['pre1'] = pred1
                test_data.to_excel(f'{config.output}/output_test.xlsx',index=False)

 