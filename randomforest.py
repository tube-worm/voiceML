#! /usr/bin/env python
# coding: utf-8

"""
construct randomforest to predict voice actress

configure [RANDOMFOREST] section in "config/config.ini"
    NUM_OF_ESTIMATORS
     
"""

import glob
import librosa
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pylab
from matplotlib.pyplot import specgram

import vmlutil

def normalization(cm):
    new_cm = []
    for line in cm:
        sum_val = sum(line)
        if not sum_val:
            sum_val = 1
        new_array = [float(num)/float(sum_val) for num in line]
        new_cm.append(new_array)
    return new_cm

def plot_confusion_matrix(cm,class_name,char_class,title):
    pylab.clf()
    #pylab.matshow(cm,fignum=False,cmap='Blues',vmin=0.0, vmax=1.0)
    pylab.matshow(cm,fignum=False,cmap='Greys')
    ax = pylab.axes()
    ax.set_xticks(range(len(class_name)))
    ax.set_xticklabels(class_name,rotation =90)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(char_class)))
    ax.set_yticklabels(char_class)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    #pylab.figure(figsize=(100, 6))
    pylab.xlabel('Predict class')
    pylab.ylabel('character class')
    pylab.grid(False)
    #pylab.tight_layout()
    outputfilename = title + ".png"
    pylab.savefig(outputfilename,bbox_inches="tight", transparent=False)

def main():
    configFilepath = "./config/config.ini"
    trainDirpath, testDirpath, train_lst, test_lst = vmlutil.read_config(configFilepath)
    n_estimators = vml.util.read_config_randomforest(configFilepath)
    x, y = vmlutil.read_mfcc(train_lst, trainDirpath)
    RF = RandomForestClassifier(n_estimators=n_estimators)
    RF.fit(x,y)
    
    predx, predy = vmlutil.read_mfcc(test_lst, testDirpath)
    prediction = RF.predict(predx)
    confusionMatrix = normalization(confusion_matrix(predy, pred))
    plot_confusion_matrix(cm, train_list, test_list, "confusion matrix")
    
if __name__ == '__main__':
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {}  sec'.format(elapse))