import gzip
import hdf5_getters

def get_data():
    with open("data/millionsongsubset.tar.gz", 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        out = bytestream.read(1000)
        print(out)

def test_thing():
    h5 = hdf5_getters.open_h5_file_read("data/millionsongsubset.tar.gz")
    duration = hdf5_getters.get_duration(h5)
    h5.close()