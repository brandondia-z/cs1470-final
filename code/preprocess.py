import gzip

def get_data():
    with open("data/millionsongsubset.tar.gz", 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        out = bytestream.read(8)
        print(out)
