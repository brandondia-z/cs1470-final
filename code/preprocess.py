import gzip
import h5py
import os
import time
import numpy as np
import json
# import librosa

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    From https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_data(start, end):
    print("Gathering data...")
    printProgressBar(0, (end-start), prefix = 'Progress:', suffix = 'Complete', length = 50)
    
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
    for dirName, subdirList, fileList in os.walk("data/MillionSongSubset"):
        for f in fileList:
            if(i>=start):
                if(f.endswith('.h5')):
                    file = h5py.File(dirName + '/' + f, 'r')
                    if(file['analysis']['segments_pitches'].shape[0]) > minlen:
                        try:
                            fat = "data/lastfm_subset" + dirName[-6:] + "/" + f[:-2] + "json"
                            jfile = open(fat)
                            jtags = json.load(jfile)['tags']
                            curtag = []
                            for tag in jtags:
                                if tag[0] in top_tags:
                                    if len(curtag)==0:
                                        curtag = tag
                                    elif top_tags[tag[0]] > top_tags[curtag[0]]:
                                        curtag = tag
                            jfile.close()
                            if len(curtag) != 0:
                                tags.append(curtag)
                                segments_timbres.append(np.array(file['analysis']['segments_timbre'][0:minlen]))
                                segments_pitches.append(np.array(file['analysis']['segments_pitches'][0:minlen]))
                        except:
                            pass
                    file.close()
                    
                    printProgressBar(i, (end-start), prefix = 'Progress:', suffix = 'Complete', length = 50)
                i+=1
                if(i>end+1):
                    tag_labels=[]
                    for tag in tags:
                        tag_labels.append(top_tags[tag[0]])
                    return (segments_timbres, segments_pitches, tag_labels)
