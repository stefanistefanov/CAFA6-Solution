import gzip
import pickle


def save_pickle(obj, filepath, compress=False):
    with gzip.open(filepath, 'wb') if compress else open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath, compress=False):
    with gzip.open(filepath, 'rb') if compress else open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj
