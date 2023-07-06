import random

import bvn
import numpy as np
import tensorflow as tf
import yaml
import glob

from config import dataConfig as dc

FULL_TO_SHORT = {
    "PController": "pc",
    "BPController": "bpc",
    "BPCPhi": "bpcphi",
    "BPCPhiT": "bpcphit",
    "CA": "ca",
    "CAHinge": "cahinge",
    "OracleController": 'oracle',
    "CP": "cp",
    "CPHinge": "cphinge",
    "CPB": "cpb",
    "CPBHinge": "cpbhinge",
    #"LPPController": "lpc",
    #"LPPNoStateController": "lpnc",
    #"SMPController": "smpc",
    #"OnlineBPController": "onlinebpc",
    #"SMPCT_IS": "smpct_is",
    #"OracleController": "oracle",
    #"BPCPhi": "bpcphi",
    #"OnlineBPNoErrorHingeController": "onlinebpcnoerrorhinge",
    #"BPCPhiT": "bpcphit",
    #"OnlineBPNoErrorProjectController": "onlinebpcprojectnoerror",
    #"SMPCT_IS_Error_Org": "smpct_is_error_org"
}

SHORT_TO_FULL = {
    "pc": "PController",
    "bpc": "BPController",
    "bpcphi": "BPCPhi",
    "bpcphit": "BPCPhiT",
    "ca": "CA",
    "cahinge": "CAHinge",
    "oracle": "OracleController",
    "cp": "CP",
    "cphinge": "CPHinge",
    "cpb": "CPB",
    "cpbhinge": "CPBHinge",
    #"lpc": "LPPController",
    #"lpnc": "LPPNoStateController",
    #'base': "BaseController",
    #"smpc": "SMPController",
    #"onlinebpc": "OnlineBPController",
    #"smpct_is": "SMPCT_IS",
    #"oracle": "OracleController",
    #"bpcphi": "BPCPhi",
    #"onlinebpcnoerrorhinge": "OnlineBPNoErrorHingeController",
    #"bpcphit": "BPCPhiT",
    #"onlinebpcprojectnoerror": "OnlineBPNoErrorProjectController",
    #"smpct_is_error_org": "SMPCT_IS_Error_Org"
}


def exposure_fn(rk):
    if dc.DATASET == "toy":
        return np.array([1, 0.5, 0])
    elif dc.DATASET == 'movie_len_top_10':
        print("dc.DATASET == 'movie_len_top_10'")
        e = 1 / rk
        e[16:] = 0
        return e
    else:
        return 1 / rk


def dcg_util_fn(rk):
    if dc.DATASET == "toy":
        return np.array([1, 0.2, 0])
    elif dc.DATASET == 'movie_len_top_10':
        print("dc.DATASET == 'movie_len_top_10'")
        dcg = 1 / np.log2(rk + 1)
        dcg[16:] = 0
        return dcg
    else:
        dcg = 1 / np.log2(rk + 1)
        return dcg

def cost_fn(c, k, m):
    return c / np.power(2, k * np.arange(m))

def to_permutation_matrix(permutation):
    """
    Convert a permutation to a permutation matrix.
    """
    n = len(permutation)
    P = np.zeros((n, n))
    for i, j in enumerate(permutation):
        P[j, i] = 1
    return P


def bvn_decomp(U):
    U = tf.convert_to_tensor(U[np.newaxis, :, :].astype(np.float32))
    p, c = bvn.bvn(U, 10)
    return list(p.numpy()[0]), list(c.numpy()[0])


def sample_relevance(n, replace=True):
    D = np.load(dc.EXP_DIR / "train.npy")[:, :dc.N]
    if n is None or dc.IS_TEMPORAL:
        return D
    #return D[np.random.choice(D.shape[0], n, replace=replace)]
    return D

def sample_sequence_relevance(n, replace=True):
    files = glob.glob(f"{dc.EXP_DIR}/sequences/*.npy")
    D = []
    for file in files:
        D.append(np.load(file))
    D = np.stack(D, axis=0)
    #D = np.load(dc.EXP_DIR / sequences / "train.npy")[:, :dc.N]
    #if n is None or dc.IS_TEMPORAL:
    #    return D
    #return D[np.random.choice(D.shape[0], n, replace=replace)]
    return D

def sample_actual_relevance(n):
    R = np.load(dc.EXP_DIR / "test.npy")[:, :dc.N]
    if n is None:
        return R
    #return R[np.random.choice(R.shape[0], n, replace=False)]
    return R

def get_test_relevance():
    R = np.load(dc.EXP_DIR / "test.npy")[:, :dc.N]
    return R

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_conf(path, targets, dev, relevance_type):
    filename = "dev.npy" if dev else "test.npy"
    estimate = np.loadtxt(dc.EXP_DIR / f"exposure_estimate_{dc.N}.txt")
    R = np.load(dc.EXP_DIR / filename)[:, :dc.N]
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    groups, _ = config["groups"], config["targets"]
    if targets is None:
        targets = config["targets"]
    else:
        config["targets"] = targets
    assert len(groups) == len(targets)

    M = np.zeros((len(groups), dc.N), dtype=np.int32)
    E = np.zeros((len(groups),), dtype=float)
    for gid, group in enumerate(groups):
        for idx in group:
            M[gid, idx] = 1
            E[gid] += estimate[idx]
    delta = np.multiply(E, np.array(targets))
    print(f'E: {E}')
    print(f'delta: {delta}')
    T = R.shape[0]
    m = M.shape[0]

    return T, R, M, delta, config, m

def ewma(x, y, alpha):
    "Exponentially weighted moving average (`x` is updated in place)."
    x[:] *= (1-alpha)
    x[:] += alpha*y
