import tensorflow as tf
from preprocess import get_data
import numpy as np

def main():
    print("Hello world!")
    timbres, pitches = get_data(0, 100)
    write_to = open('data/processed.txt', 'w')
    write_to.write(str(timbres))
    write_to.close()
    return 1


if __name__ == "__main__":
    main()