import sys
import os
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import word2ball


def get_all_functions():
    print([a for a in dir(word2ball) if isinstance(word2ball.__dict__.get(a), types.FunctionType)])


if __name__ == '__main__':
    pass
