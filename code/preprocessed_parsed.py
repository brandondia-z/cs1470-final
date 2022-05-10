import gzip
import h5py
import os
import time
import numpy as np
import json
import pickle

def get_parsed(i):
    with open (f'data/parsed_data/pitches{i}', 'rb') as fp:
        pitches = pickle.load(fp)
    with open (f'data/parsed_data/ftags{i}', 'rb') as fp:
        tags = pickle.load(fp)
    with open (f'data/parsed_data/timbres{i}', 'rb') as fp:
        timbres = pickle.load(fp)
    inputs = np.concatenate((timbres, pitches), axis=2)
    return (inputs, tags)

def fix_parsed(i):
    top_tags = {}
    tag_id = 0
    with open("data/top_tags.txt") as file:
        for line in file:
            tokens = line.split()
            top_tags[' '.join(tokens[0:-1])] = tag_id
            tag_id+=1
    with open (f'data/parsed_data/tags{i}', 'rb') as fp:
            tags = pickle.load(fp)
    tag_labels=[]
    for tag in tags:
        tag_labels.append(top_tags[tag[0]])
    with open (f'data/parsed_data/ftags{i}', 'wb') as fp:
        pickle.dump(tag_labels, fp)