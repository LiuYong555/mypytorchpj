# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02:49 2022

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
#Onehot-coding函数：对原始样本编码
onehotcoding=Onehot_coding(length=50)
X_embedding=torch.tensor(onehotcoding.transform(X_data))
X_train,X_test,y_train,y_test=train_test_split(X_embedding, y,
              test_size=0.2, random_state=42, stratify=y) 

#组织数据集：训练集和测试集
train_data=Text_Data(X_train,y_train)
train_loader=DataLoader(train_data,batch_size=100,shuffle=True)
test_data=Text_Data(X_test,y_test)
test_loader=DataLoader(test_data,batch_size=100,shuffle=False)
#设置超参数
embedding_size = 5 # embedding size
sequence_length = 50 # sequence length
num_classes = 5 # number of classes
filter_sizes = [2, 3, 4] # n-gram windows
num_filters = 50 # number of filters
vocab_size=len(onehotcoding.vocab)+1 #+1代表填充值的编码



###randomCNN：迭代训练
model=TextCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_record=[]
correct_record=[]
for epoch in range(1,1000+1):
    loss_train=0.0
    correct_num=0.0
    for batch in train_loader:
        inputs,labels=batch
        outputs=model(inputs)

        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_train+=loss.item()
        pred=outputs.argmax(dim=1)
        correct_num+=(pred.reshape(labels.shape)==labels).sum().item()

    if epoch==1 or epoch%50==0:
        correct_record.append(correct_num/len(X_train))
        loss_record.append(loss_train/len(data_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(),epoch,
            loss_train/len(data_loader)))

###randomCNN:绘图
randomCNN_loss=loss_record
randomCNN_correct=correct_record
Plotmodel(randomCNN_loss,randomCNN_correct)

###randomCNN:计算准确率
validate(model,train_loader,test_loader)


###randomRNN：迭代训练
batch_size=100
hidden_size=10
num_class=5
num_layers=3
sequence_length = 50 
model=TextRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_record=[]
correct_record=[]
for epoch in range(1,1000+1):
    loss_train=0.0
    correct_num=0.0
    for batch in train_loader:
        inputs,labels=batch
       
        outputs=model(inputs)

        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_train+=loss.item()
        pred=outputs.argmax(dim=1)
        correct_num+=(pred.reshape(labels.shape)==labels).sum().item()

    if epoch==1 or epoch%50==0:
        correct_record.append(correct_num/len(X_train))
        loss_record.append(loss_train/len(train_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(),epoch,
            loss_train/len(data_loader)))

###randomRNN:绘图
randomRNN_loss=loss_record
randomRNN_correct=correct_record
Plotmodel(randomRNN_loss,randomRNN_correct)

###randomRNN:计算准确率
validate(model,train_loader,test_loader)

















