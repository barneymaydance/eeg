#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/15/16 14:53
# @Author  : Sheng Lin
# @Site    :
# @File    : save.py


import csv
import numpy
import marshal as m
from datetime import datetime

def readRawData(fileno, select=True):
    filename = "indra_mids_5_15/neurosky-data/" + str(fileno) + ".csv"
    data=[]
    startTime1 =datetime.strptime("23:33:28","%H:%M:%S")
    endTime1 =datetime.strptime("23:33:58","%H:%M:%S")
    startTime2 =datetime.strptime("23:44:26","%H:%M:%S")
    endTime2 =datetime.strptime("23:44:56","%H:%M:%S")
    # startTime1 =datetime.strptime("23:33:28.876","%H:%M:%S")
    # endTime1 =datetime.strptime("23:33:58.875","%H:%M:%S")
    # startTime2 =datetime.strptime("23:44:26.341","%H:%M:%S")
    # endTime2 =datetime.strptime("23:44:56.343","%H:%M:%S")
    with open(filename,"r") as csvfile:
        reader = csv.reader(csvfile)
        for index, col in enumerate(reader):
            if index==0:
                continue
            if(select):
                data.append([int(i) for i in col[7][1:-1].split(",")])
            else:
                time=col[1].split(" ")[1]
                time=time[:time.find("+")]
                time = time[:time.find(".")]
                t=datetime.strptime(time,"%H:%M:%S")
                if(t>=startTime1 and t<=endTime1) or (t>=startTime2 and t<=endTime2):
                    data.append([int(i) for i in col[7][1:-1].split(",")])
    return data

for person in range(1,31):
    data=readRawData(person, False)
    print len(data)
    m.dump(data[:400],open("input/person_select_data"+str(person)+".p","w"))

# data=readRawData(1)
#
# power_data=[]
# label_data=[]
#
#
#
# fft=numpy.fft.fft(data[0])
# print(fft)
# psd=numpy.abs(fft)**2
# print (len(fft))
# print psd
