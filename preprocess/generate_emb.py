import sys
import pickle

import numpy as np
from xclib.data import data_utils

from config import dataConfig as dc


def generate_emb(raw_feature, embeddings, indices=None):
    vec = dict()
    if indices is None:
        indices = list(range(raw_feature.shape[0]))
    for index in indices:
        row = raw_feature[index]
        embs = []
        _, cols = row.nonzero()
        for col in cols:
            embs.append(embeddings[col])
        vec[index] = np.mean(embs, axis=0)
    return vec


if __name__ == "__main__":
    dc.init_config(sys.argv[1])
    DATA_DIR, EXP_DIR = dc.DATA_DIR, dc.EXP_DIR

    embeddings_path = DATA_DIR / "fasttextB_embeddings_512d.npy"
    embeddings = np.load(embeddings_path)

    print("Generating query embeddings in train set for MLP...")
    train_path = DATA_DIR / "test.txt"
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        train_path
    )
    train_query_features = features
    train_query_ids = range(10000)
    train_query_emb = generate_emb(
        train_query_features, embeddings, train_query_ids)
    pickle.dump(train_query_emb, open(EXP_DIR / "train_query_emb.pkl", "wb"))

    print("Generating doc embedding...")
    doc_features = data_utils.read_sparse_file(
        DATA_DIR / "Yf.txt", header=True)
    doc_ids = np.loadtxt(EXP_DIR / "doc_ids.txt", dtype=int)
    doc_emb = generate_emb(doc_features, embeddings, doc_ids)
    pickle.dump(doc_emb, open(EXP_DIR / "doc_emb.pkl", "wb"))

    print("Generating query embedding for similation experiment...")
    train_path = DATA_DIR / "train.txt"
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        train_path)
    query_features = features
    query_ids = np.loadtxt(EXP_DIR / "query_ids.txt", dtype=int)
    query_emb = generate_emb(query_features, embeddings, query_ids)
    pickle.dump(query_emb, open(EXP_DIR / "query_emb.pkl", "wb"))
