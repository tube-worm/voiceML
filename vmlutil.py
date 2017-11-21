# conding: utf-8

import os
import sys
import numpy as np
from configparser import ConfigParser

TRAINDIR = None
TESTDIR = None

def set_config(configFilepath):
    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)
    return config

def read_config(configFilepath):
    config = set_config(configFilepath)
    trainDirpath = config["DEFAULT"]["TRAINDIR"]
    testDirpath  = config["DEFAULT"]["TESTDIR"]

    namelistFilename = open(os.path.join(trainDirpath, "namelist"))
    train_lst = namelistFilename.read().split('\n')
    namelistFilename.close()
    train_lst = train_lst[:len(train_lst)-1]
    
    namelistFilename = open(os.path.join(testDirpath, "namelist"))
    test_lst = namelistFilename.read().split('\n')
    namelistFilename.close()
    test_lst = test_lst[:len(test_lst)-1]
    
    return trainDirpath, testDirpath, train_lst, test_lst

def read_config_mfcc(configFilepath):
    config = set_config(configFilepath)
    samplingRate = int(config["MFCC"]["SAMPLING_RATE"])
    frameLength = int(config["MFCC"]["FRAME_LENGTH"])
    hopLength = int(config["MFCC"]["HOP_LENGTH"])
    return samplingRate, frameLength, hopLength
    
def delta_cepstrum(mfcc):
    x = np.array((1,2,3,4,5))
    y = mfcc
    lm = linear_model.LinearRegression()
    lm.fit(x[:,np.newaxis],y[:,np.newaxis])
    return lm.coef_[0]

def read_mfcc(name_list, base_dir, n_sample):
    x,y = [],[]
    for label, name in enumerate(name_list):
        for filename in glob.glob(os.path.join(base_dir, name, "*.mfcc.npy")):
            mfccs = np.load(filename)
            num_mfccs = len(mfccs)
            for samplingnum in range(NUM_OF_SAMPLING):
                frame_index = np.random.randint(num_mfccs)
                dc_vector = []
                for i in range(len(mfccs[frame_index])):
                    dc = delta_cepstrum(mfccs[frame_index][i])
                    dc_vector.append(dc[0])
                x.append(np.c_[mfccs[frame_index], dc_vector])
                y.append(label)
    return np.array(x), np.array(y)

def calc_mfccs_features(mfccs):
    def calc_mfcc_features(mfcc_seq):
        deltaCepstrum_vector = []
        for k in range(len(mfcc_seq) - 4):
            deltaCepstrum = delta_cepstrum(mfcc_seq[k:k+5])
            deltaCepstrum_vector.append(dc[0])
        deltadelta = delta_cepstrum(np.array(deltaCepstrum_vector))
        return np.r_[dc_vector, deltadelta]
    return np.c_[mfccs, np.array(list(map(calc_mfcc_features,mfccs)))]

def read_mfcc_all(name_list, base_dir, n_sample):
    x,y = [],[]
    for label, name in enumerate(name_list):
        filename = os.path.join(base_dir, name, (name + ".wav.mfcc.npy"))
        mfccs = np.load(filename)
        features = list(map(feature_vector, mfccs))
        x.extend(features)
        y.extend([label] * len(features))
    return np.array(x), np.array(y)

