import gzip
import h5py

def get_data():
    f = h5py.File('data/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5', 'r')
    print(list(f.keys()))