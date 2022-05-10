import gzip
import h5py
import os
import time
import numpy as np
import json
# import librosa

def get_count():
    print("counting files")
    count = 0
    for dirName, subdirList, fileList in os.walk("..\dl-data\data"):
        for f in fileList:
            count+=1
    print(count)

if __name__ == '__main__':
    print(get_count())