import gzip
import h5py
import librosa

def get_data():
    f = h5py.File('data/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
    print(list(f.keys()))
    print(list(f['analysis'].keys()))
    print(list(f['analysis']['segments_pitches']))
    