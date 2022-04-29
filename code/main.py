import tensorflow as tf
from preprocess import get_data
import numpy as np

def main():
    print("Hello world!")
    timbres, pitches = get_data(0, 100)
    np.savetxt('data/processed.txt', timbres)
    return 1


if __name__ == "__main__":
    main()