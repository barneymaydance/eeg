#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/15/16 14:53
# @Author  : Sheng Lin
# @Site    : 
# @File    : main.py

import numpy
import marshal as m
from scipy.spatial.distance import cdist
from RSM import rsm

male=[1, 3, 8, 9, 22, 24, 28]
female=[2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 29, 30]


raw_data=[]
train_label_data=[]
test_label_data=[]
train_data=[]
test_data=[]
for person in range(1,31):
    raw_data.append(m.load(open("input/person_select_data"+str(person)+".p","r")))


print (len(raw_data))
print (len(raw_data[0]))
num_of_total=30
num_of_train=num_of_total*2/3
num_of_test=num_of_total-num_of_train

num_of_person=30
# print num_of_train
# print num_of_test


for j in range(num_of_train):
    for i in range(num_of_person):
        train_data.append(raw_data[i][j])
        train_label_data.append(i)
for j in range(num_of_train,num_of_total):
    for i in range(num_of_person):
        test_data.append(raw_data[i][j])
        test_label_data.append(i)

# print(len(train_label_data))
# print(len(train_data))
# print(len(test_label_data))
# print(len(test_data))

def normalizing(raw_data):
    smallest=0
    for e in raw_data:
        smallest=min(smallest,min(e))
    new_data=numpy.array(raw_data)
    new_data=new_data-smallest
    return new_data

def fft_normalizing(raw_data):
    new_data=[]
    for data in raw_data:
        fft=numpy.fft.fft(data)
        psd=numpy.abs(fft)**2
        new_data.append(numpy.log10(psd+1).astype(numpy.int64))
    new_data = numpy.array(new_data)
    return new_data

train_data=fft_normalizing(train_data)
test_data=fft_normalizing(test_data)

model=rsm.RSM(train_data,num_visible=512,num_hidden=128)
model.train(max_epochs=25, batch_size=50, step=1)
correct=0
c_gender=0
for test in range(len(test_data)):
    ind=model.recommendByTraindata(test_data[test], train_data, Rank=1)
    # print ("returned index: {}".format(ind))
    # print ("test case label: {}".format(test_label_data[test]))
    label_list=[train_label_data[i] for i in ind]
    # print ("returned label list: {}".format(label_list))
    if test_label_data[test] in label_list:
        correct+=1
    if test_label_data[test] in female and label_list[0] in female:
        c_gender+=1
    elif test_label_data[test] in male and label_list[0] in male:
        c_gender+=1

print correct
print correct*1.0/len(test_data)
print c_gender
print c_gender*1.0/len(test_data)

