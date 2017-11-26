#! /usr/bin/env python
# coding: utf-8

"""
calculate mfcc for each .wav file
dump numpy files of mfcc

configure [MFCC] section in "config/config.ini"
    SAMPLING_RATE
    FRAME_LENGTH
    HOP_LENGTH
"""

import glob
import librosa
import numpy as np
import os
import time

import vmlutil

def create_mfcc(wavFilename, samplingRate, frameLength, hopLength):
    wave, samplingRate = librosa.load(wavFilename, sr=samplingRate)
    splitTime_arr = librosa.effects.split(wave)   # split wave signal into non-silent interval

    concatenatedWave = []                         # non-silent wave signal
    for interval in splitTime_arr:                # interval: list of start([0]) and end([1])
        trimmedWave , index = librosa.effects.trim(wave[interval[0]:interval[1]])
        concatenatedWave = np.r_[concatenatedWave, trimmedWave]
    
    frameWave_arr = librosa.util.frame(concatenatedWave, frame_length=frameLength, hop_length=hopLength)
    
    mfcc_lst = []
    for i in range(frameWave_arr.shape[1]):
        mfcc = librosa.feature.mfcc(y=frameWave_arr[:, i], sr=samplingRate)
        mfcc_lst.append(mfcc[:])
    mfccFilename = wavFilename + ".mfcc"
    np.save(mfccFilename, mfcc_lst)

def main():
    configFilepath = "./config/config.ini"
    trainDirpath, testDirpath, train_lst, test_lst = vmlutil.read_config(configFilepath)
    samplingRate, frameLength, hopLength = vmlutil.read_config_mfcc(configFilepath)
    for wavFilename in glob.glob(trainDirpath + "*/*.wav"):
        print("START: calc mfcc of {}".format(wavFilename))
        create_mfcc(wavFilename, samplingRate=samplingRate, frameLength=frameLength, hopLength=hopLength)
        print("DONE: dump mfcc to {}.mfcc".format(wavFilename))
    for wavFilename in glob.glob(testDirpath + "*/*.wav"):
        print("START: calc mfcc of {}".format(wavFilename))
        create_mfcc(wavFilename, samplingRate=samplingRate, frameLength=frameLength, hopLength=hopLength)
        print("DONE: dump mfcc to {}.mfcc".format(wavFilename))
    
if __name__ == '__main__':
    start = time.time()
    main()
    elapse = time.time() - start
    print('\nelapse time: {}  sec'.format(elapse))