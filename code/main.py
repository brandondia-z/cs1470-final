import tensorflow as tf
from preprocess import get_data
import numpy as np

def main():
    print("Hello world!")
    timbres, pitches = get_data(0, 100)
    # print(np.asarray(timbres).shape)
    return 1


if __name__ == "__main__":
    main()