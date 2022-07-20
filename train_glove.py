# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:52:01 2022

@author: 刘永
"""

import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
os.chdir('E:\pythonfile\datascience')
from model import *
import datetime
###数据准备

df=pd.read_csv("sentiment-analysis-on-movie-reviews/train.tsv/train.tsv",
               encoding='gbk', sep='\t')
df['length']=df['Phrase'].apply(lambda x:len(x.strip().split(' ')))
df=df[(df['length']>=20)&(df['length']<=50)]         #对原始数据做简单筛选
df=df.head(10000)
X_data, y_data =df["Phrase"].values,df["Sentiment"].values
y = np.array(y_data).reshape((-1, 1))

###
#准备预训练好的gloveembedding
with open('E:\pythonfile\glove.6B.50d.txt',encoding='utf-8') as file:
    content=file.readlines()
glove_dict={}
for i in range(len(content)):
    str_list=content[i].rstrip().split(' ')[1:]
    weight_list=[float(s) for s in str_list]
    glove_dict[content[i].rstrip().split(' ')[0]]=weight_list
    
X_train,X_test,y_train,y_test=train_test_split(X_data, y,
                    test_size=0.2, random_state=42, stratify=y)
#训练集
glove_train=[]
for sentence in X_train:
    sentence_feature=[]
    word_list=sentence.strip().split(' ')
    for word in word_list:
        if(word in glove_dict.keys()):
            sentence_feature.append(glove_dict[word])
        else:
            sentence_feature.append([0]*50)
    for i in range(50-len(sentence_feature)):
        sentence_feature.append([0]*50)
    t_sen_feature=torch.tensor(sentence_feature)
    glove_train.append(t_sen_feature)
#测试集
glove_test=[]
for sentence in X_test:
    sentence_feature=[]
    word_list=sentence.strip().split(' ')
    for word in word_list:
        if(word in glove_dict.keys()):
            sentence_feature.append(glove_dict[word])
        else:
            sentence_feature.append([0]*50)
    for i in range(50-len(sentence_feature)):
        sentence_feature.append([0]*50)
    t_sen_feature=torch.tensor(sentence_feature)
    glove_test.append(t_sen_feature)
    
#准备glove初始化得到的数据
glove_data=torch.stack(([sen for sen in glove_train]),dim=0)
glove_test_x_data=torch.stack(([sen for sen in glove_test]),dim=0)
glove_train_data=Text_Data(glove_data,y_train)
glove_train_loader=DataLoader(glove_train_data,batch_size=100,shuffle=True)
glove_test_data=Text_Data(glove_test_x_data,y_test)
glove_test_loader=DataLoader(glove_test_data,batch_size=100,shuffle=True)

#gloveCNN:迭代训练
model=GloveTextCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_record=[]
correct_record=[]
for epoch in range(1,1000+1):
    loss_train=0.0
    correct_num=0.0
    for batch in glove_train_loader:
        inputs,labels=batch
        outputs=model(inputs)

        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_train+=loss.item()
        pred=outputs.argmax(dim=1)
        correct_num+=(pred.reshape(labels.shape)==labels).sum().item()

    if epoch==1 or epoch%50==0:
        correct_record.append(correct_num/len(glove_train_loader))
        loss_record.append(loss_train/len(glove_train_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(),epoch,
            loss_train/len(glove_train_loader)))

###gloveCNN:绘图
gloveCNN_loss=loss_record
gloveCNN_correct=correct_record
Plotmodel(gloveCNN_loss,gloveCNN_correct)

###randomCNN:计算准确率
validate(model,glove_train_loader,glove_test_loader)























