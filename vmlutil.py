# conding: utf-8

import sys
import numpy as np
import configparser import ConfigParser

def set_config(configFilepath):
    try:
        config = ConfigParser()
        config.read(configFilepath)
    except FileNotFoundError:
        print("Not Found: {}".format(configFilepath))
        sys.exit(1)
    return config

def read_config(configFilepath):
    global TRAINDIR
    global TESTDIR
    
    config = set_config(configFilepath)
    TRAINDIR = config["DEFAULT"]["TRAINDIR"]
    TESTDIR  = config["DEFAULT"]["TESTDIR"]