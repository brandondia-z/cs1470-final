import tensorflow as tf
from preprocess import get_data, test_thing

def main():
    print("Hello world!")
    get_data()
    test_thing()
    return 1


if __name__ == "__main__":
    main()