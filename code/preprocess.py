import gzip
import h5py
import os
import time
import numpy as np
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
    f = h5py.File('data/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
    # print(list(f.keys()))
    # print(list(f['analysis'].keys()))
    # print(f['analysis']['segments_pitches'].shape)
    # print(f['analysis']['segments_timbre'].shape)
    segments_timbres = []
    segments_pitches = []
    i = 0
    for dirName, subdirList, fileList in os.walk("data/MillionSongSubset"):
        for f in fileList:
            if(i>=start):
                if(f.endswith('.h5')):
                    file = h5py.File(dirName + '/' + f, 'r')
                    # print(np.array(file['analysis']['segments_timbre'][:]).shape)
                    # time.sleep(5)
                    segments_timbres.append(np.array(file['analysis']['segments_timbre'][:]))
                    segments_pitches.append(file['analysis']['segments_pitches'])
                    file.close()
                printProgressBar(i + 1, (end-start), prefix = 'Progress:', suffix = 'Complete', length = 50)
                i+=1
                if(i>end):
                    print(np.array(segments_timbres[0]).shape)
                    return (segments_timbres, segments_pitches)