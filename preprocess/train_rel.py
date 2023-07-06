import sys
import pickle
import random

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from xclib.data import data_utils

from config import dataConfig as dc

if __name__ == "__main__":
    dc.init_config("xc")
    DATA_DIR, EXP_DIR = dc.DATA_DIR, dc.EXP_DIR

    random.seed(0)
    np.random.seed(0)

    train_path = DATA_DIR / "test.txt"

    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        train_path
    )
    doc_emb = pickle.load(open(EXP_DIR / "doc_emb.pkl", "rb"))
    train_query_emb = pickle.load(open(EXP_DIR / "train_query_emb.pkl", "rb"))

    X, y = [], []
    pos, neg = 0, 0
    for query_id in train_query_emb:
        for doc_id in doc_emb:
            feature = np.concatenate(
                (train_query_emb[query_id], doc_emb[doc_id]), axis=None
            )
            if labels[query_id, doc_id]:
                X.append(feature)
                y.append(1)
                pos += 1
            else:
                if random.random() < 0.0005:
                    X.append(feature)
                    y.append(0)
                    neg += 1
    X = np.array(X)
    y = np.array(y)

    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000).fit(X, y)
    pickle.dump(clf, open(EXP_DIR / "mlp.model", "wb"))

    query_emb = pickle.load(open(EXP_DIR / "query_emb.pkl", "rb"))
    doc_ids = np.loadtxt(EXP_DIR / "doc_ids.txt", dtype=int)
    query_ids = np.loadtxt(EXP_DIR / "query_ids.txt", dtype=int)

    def get_relevance(q_emb):
        X = []
        for doc_id in doc_ids:
            d_emb = doc_emb[doc_id]
            X.append(np.concatenate((q_emb, d_emb), axis=None))
        X = np.array(X)
        rel = clf.predict_proba(X)[:, 1].flatten()
        return rel

    D_ids, R_ids = train_test_split(query_ids, test_size=0.2, random_state=0)
    D = np.zeros((len(D_ids), len(doc_ids)))
    R = np.zeros((len(R_ids), len(doc_ids)))

    for i, query_id in enumerate(D_ids):
        D[i, :] = get_relevance(query_emb[query_id])
    np.save(EXP_DIR / "D.npy", D)

    for i, query_id in enumerate(R_ids):
        R[i, :] = get_relevance(query_emb[query_id])
    np.save(EXP_DIR / "R.npy", R)
