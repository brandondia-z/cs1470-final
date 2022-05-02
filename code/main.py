import tensorflow as tf
from preprocess import get_data
import numpy as np

def main():
    print("Hello world!")
    timbres, pitches, tags = get_data(0, 10000)
    return 1


if __name__ == "__main__":
    main()
