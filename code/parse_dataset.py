import gzip
import h5py
import os
import time
import numpy as np
import json
import sqlite3
import pickle

def parse_data():
    print("Parsing data from dataset")
    
    # Get top 50 tags in last fm
    top_tags = {}
    tag_id = 0
    with open("data/top_tags.txt") as file:
        for line in file:
            tokens = line.split()
            top_tags[' '.join(tokens[0:-1])] = tag_id
            tag_id+=1

    segments_timbres = []
    segments_pitches = []
    tags = []
    i = 0
    num = 0
    minlen = 200
    tot_songs = 0

    conn = sqlite3.connect("data/lastfm_tags.db")

    for dirName, subdirList, fileList in os.walk("..\dl-data\data"):
        for f in fileList:
            file = h5py.File(dirName + '/' + f, 'r')
            if(file['analysis']['segments_pitches'].shape[0] > minlen):
                sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid='%s'" % f[:-3]
                res = conn.execute(sql)
                data = res.fetchall()
                curtag = []
                for tag in data:
                    if tag[0] in top_tags:
                        if len(curtag)==0:
                            curtag = tag
                        elif top_tags[tag[0]] > top_tags[curtag[0]]:
                            curtag = tag
                if len(curtag) != 0:
                    tags.append(curtag)
                    segments_timbres.append(np.array(file['analysis']['segments_timbre'][0:minlen]))
                    segments_pitches.append(np.array(file['analysis']['segments_pitches'][0:minlen]))
                    if len(tags) > 10000:
                        with open(f'data/parsed_data/tags{i}', 'wb') as fp:
                            pickle.dump(tags, fp)
                        with open(f'data/parsed_data/timbres{i}', 'wb') as fp:
                            pickle.dump(segments_timbres, fp)
                        with open(f'data/parsed_data/pitches{i}', 'wb') as fp:
                            pickle.dump(segments_pitches, fp)
                        tags = []
                        segments_timbres = []
                        segments_pitches = []
                        print(i)
                        i+=1

            file.close()


if __name__ == '__main__':
    parse_data()