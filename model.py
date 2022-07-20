# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:08:20 2022

@author: 刘永
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import numpy as np

###原始数据one-hot编码
class Onehot_coding:
    def __init__(self,length):
        self.vocab={}
        self.length=length
        #self.num_
    def transform(self,sent_list):
        embedding_features=[]
        for sent in sent_list:
            sent=sent.lower()
            words=sent.strip().split(' ')
            for word in words:
                if word not in self.vocab:
                    self.vocab[word]=len(self.vocab)+1
        for sent in sent_list:
            sent=sent.lower()
            words=sent.strip().split(' ')
            embedding_features.append([self.vocab[word] for word in words])
        for item in embedding_features:
            item=item.extend([0]*(self.length-len(item)))
        return embedding_features

###组织数据集
class Text_Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index, :] 



###绘图函数
#单个模型的训练集上损失函数与准确率
def Plotmodel(loss,correct):
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(111)
    ax2=ax1.twinx()
    epoch_label=[1]
    for i in range(1,21):
        epoch_label.append(50*i)
    loss_train=np.array(loss)
    loss_train_plot=ax1.plot(epoch_label,loss_train,c='blue',linewidth=0.5)
    rate_train=np.array(correct)
    rate_train_plot=ax2.plot(epoch_label,rate_train,c='red',linewidth=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Correct Rate')

    lines=loss_train_plot+rate_train_plot
    labels=['Loss','Correct Rate']
    plt.title('Loss&Correct')
    plt.legend(lines,labels)
    fig.show()
    
###准确率计算
def validate(model,train_loader,val_loader):
    for name,loader in [('train',train_loader),('val',val_loader)]:
        correct=0
        total=0
        
        with torch.no_grad():
            for inputs,labels in loader:
                outputs=model(inputs)
                pred=outputs.argmax(dim=1)
                total+=labels.shape[0]
                correct+=(pred.reshape(labels.shape)==labels).sum().item()
                #print(correct/total)
        print('Accuracy {}:{:.2f}'.format(name,correct/total))

###神经网络模型
#1.随机初始化+CNN
embedding_size = 5 # embedding size
sequence_length = 50 # sequence length
num_classes = 5 # number of classes
filter_sizes = [2, 3, 4] # n-gram windows
num_filters = 50 # number of filters
vocab_size=12262
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X) 
        embedded_chars = embedded_chars.unsqueeze(1) 

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            
            h = F.relu(conv(embedded_chars))
            
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
        return model


#2.随机初始化+RNN
batch_size=100
hidden_size=10
num_class=5
num_layers=3
sequence_length = 50 
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN,self).__init__()
        self.embedding= nn.Embedding(vocab_size, embedding_size)
        self.rnn=nn.RNN(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*sequence_length,num_class,bias=True)
        self.h_0=torch.randn(num_layers,100,hidden_size)
    def forward(self,x):
        output=self.embedding(x)

        output=F.relu(output)
        output,_=self.rnn(output,self.h_0)
        output=output.reshape(batch_size,-1)
        output=F.relu(output)
        output=self.fc(output)
        return output



#3.glove初始化+CNN
glove_dim=50
class GloveTextCNN(nn.Module):
    def __init__(self):
        super(GloveTextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        #self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, glove_dim)) for size in filter_sizes])
    def forward(self, X):
        #embedded_chars = self.W(X) # [batch_size, sequence_length, sequence_length]
        embedded_chars = X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        #print(embedded_chars)
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
        return model


#4.glove初始化+RNN
class GloveTextRNN(nn.Module):
    def __init__(self):
        super(GloveTextRNN,self).__init__()
        #self.embedding= nn.Embedding(vocab_size, embedding_size)
        self.rnn=nn.RNN(input_size=50,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*sequence_length,num_class,bias=True)
        self.h_0=torch.randn(num_layers,100,hidden_size)
    def forward(self,x):
        output=x

        #output=F.relu(output)
        output,_=self.rnn(output,self.h_0)
        output=output.reshape(batch_size,-1)
        output=F.relu(output)
        output=self.fc(output)
        return output


















