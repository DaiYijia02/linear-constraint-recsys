import random

import torch

import cvxpy as cp
import numpy as np
from loguru import logger
from birkhoff import TOLERANCE
from sinkhorn_knopp import sinkhorn_knopp as skp
from config import dataConfig as dc
from utils import (
    bvn_decomp,
    dcg_util_fn,
    exposure_fn,
    sample_relevance,
    sample_sequence_relevance,
    sample_actual_relevance,
    get_test_relevance,
)
import multiprocessing as mp
from functools import partial
import pdb
import sys


class BaseController(object):
    def __init__(self, T, N, M, delta, config):
        self.T, self.N = T, N
        self.M = M
        self.m = M.shape[0]
        self.delta = delta
        self.u = dcg_util_fn(np.arange(N) + 1)
        self.v = exposure_fn(np.arange(N) + 1)
        self.C = config.get("C", [1.0] * M.shape[0])

        # CP class variables
        self.train_size = config.get("train_size", None)
        self.realtime_delta = delta
        self.realtime_T = T
        self.delta_ = delta
        self.T_ = T

    def get_observation(self, ranking):
        rank = np.zeros(self.N, dtype=np.int32)
        rank[ranking] = np.arange(self.N)
        return self.v[rank]

    def get_utility(self, ranking, r):
        return self.u.dot(r[ranking])

    # (state, r, tau)
    def get_action(self, state, r, tau):
        # raise NotImplementedError
        ranking_ = np.argsort(r)[::-1]
        return ranking_, "optimal"

    def act(self, state, r, tau):
        ranking, _ = self.get_action(state, r, tau)
        obs = self.get_observation(ranking)
        util = self.get_utility(ranking, r)
        state += self.M.dot(obs)
        return state, util, obs, self.M.dot(obs)


class BasePController(BaseController):
    def __init__(self, T, N, M, delta, config, lmbda):
        self.lmbda = lmbda
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def get_action(self, state, r, tau):
        # No intervention
        ranking_ = np.argsort(r)[::-1]
        obs_ = self.get_observation(ranking_)
        state_ = state + self.M.dot(obs_)
        setpoint = (tau + 1) / self.T * self.delta

        group_error = (
            self.lmbda * np.clip(setpoint - state_, 0, None) / np.sum(self.M, axis=1)
        )
        item_error = self.M.T.dot(group_error)

        s = r + item_error
        ranking = np.argsort(s)[::-1]
        return ranking, "optimal"


#class PController(BasePController):
#    def __init__(self, T, N, M, delta, config):
#        super().__init__(T, N, M, delta, config)


class BiostochasticController(BaseController):
    def solve(self):
        pass

    # TODO: remove the below function and enable the next function
    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        util = r.T @ U @ self.u
        exposure = self.M @ U @ self.v
        state += self.M.dot(obs)
        return state, util, obs, exposure

    def get_action(self, state, r):
        U = self.solve(state, r)
        permutations, coefficients = bvn_decomp(U)
        permutation = random.choices(permutations, weights=coefficients)[0]
        _, ranking = np.nonzero(permutation.T)
        return ranking


class BPController(BiostochasticController):
    pass
#    def __init__(self, T, N, M, delta, config):
#        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
#        self.hinge_min = config.get("hinge_min", 0.0)
#        super().__init__(T, N, M, delta, config)
#
#    def solve(self, state, r, tau):
#        M = cp.Constant(self.M)
#        v = cp.Constant(self.v)
#        r = cp.Constant(r)
#        u = cp.Constant(self.u)
#        U = cp.Variable((self.N, self.N), nonneg=True)
#
#        target_ = cp.Constant((tau + 1) / self.T * self.delta)
#        state_ = cp.Constant(state) + M @ U @ v
#        obj = r.T @ U @ u
#
#        ones = cp.Constant(np.ones((self.N,)))
#        constraints = [U @ ones == ones, ones @ U == ones, state_ >= target_]
#
#        prob = cp.Problem(cp.Maximize(obj), constraints)
#        prob.solve(cp.MOSEK)
#        return self.sk.fit(U.value)

class BPCPhi(BPController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.hinge_min = config.get("hinge_min", 0.0)
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        obj = r.T @ U @ u  -  C @ cp.maximum([self.hinge_min] * self.M.shape[0], (target_ - state_))

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)


class BPCPhiT(BPController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant((tau + 1) / self.T * np.array(self.C))

        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        obj = r.T @ U @ u  -  C @ cp.maximum([self.hinge_min] * self.M.shape[0], (target_ - state_))

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class CAController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        #self.momentum = config.get("momentum", -np.inf)
        self.hinge_min = config.get("hinge_min", 0.0)
        self.beta = config.get("beta", 0.5)
        self.eps = config.get("eps", 1e-5)
        super().__init__(T, N, M, delta, config)

        self.C = np.array(config.get("C", np.ones(self.m)))

        self.lr = config.get("bpc_lr", 0.01)
        self.init = config.get("bpc_init", "one")
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        if self.init == "zero":
            self.lmbda = torch.zeros((self.m,), requires_grad=True)
        elif self.init == "one":
            self.lmbda = torch.ones((self.m,), requires_grad=True)
        #self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr)
        self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr, betas=(self.beta, .999), eps=self.eps)
        self.delta_ = delta

    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        exposure = self.M @ U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        self.solve_lmbda(self.M @ obs, tau)
        return state, util, obs, exposure


    def solve_lmbda(self, state, tau):
        target_ = (1 / self.T) * self.delta_
        g = target_ - state

        # Gradient Ascent (-1)
        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(-1 * g).float()
        self.lmbda_opt.step()

        with torch.no_grad():
            min_ = torch.zeros(self.m)
            #min_ = torch.tensor([self.hinge_min] * self.M.shape[0])
            max_ = torch.tensor(self.C)
            self.lmbda.data = torch.clamp(self.lmbda.data, min=min_, max=max_).float()

class CA(CAController):
    def __init__(self, T, N, M, delta, config):
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        C = cp.Constant(self.C)
        U = cp.Variable((self.N, self.N), nonneg=True)
        #C = cp.Constant(self.C)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        #lmbda = np.clip(lmbda, np.zeros(self.m), self.C)
        lmbda = cp.Constant(lmbda)

        state_ = M @ U @ v
        obj = (r.T @ U @ u) + (lmbda @ state_)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)

class CAHinge(CAController):
    def __init__(self, T, N, M, delta, config):
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        C = cp.Constant(self.C)
        U = cp.Variable((self.N, self.N), nonneg=True)
        #C = cp.Constant(self.C)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        #lmbda = np.clip(lmbda, np.zeros(self.m), self.C)
        lmbda = cp.Constant(lmbda)

        state_ = M @ U @ v
        target_ = cp.Constant( (1. / self.T) * self.delta_)

        obj = (r.T @ U @ u) - lmbda @ cp.maximum([self.hinge_min] * self.M.shape[0], (target_ - state_))

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)

#class LPPNoStateController(BiostochasticController):
#    def __init__(self, T, N, M, delta, config):
#        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
#        self.gamma = config.get("gamma", 1)
#        self.C = config.get("C", [1.0] * M.shape[0])
#        super().__init__(T, N, M, delta, config)
#
#    def solve(self, state, r, tau):
#        r = cp.Constant(r)
#        u = cp.Constant(self.u)
#        U = cp.Variable((self.N, self.N), nonneg=True)
#
#        # Intervention
#        target_ = cp.Constant((tau + 1) / self.T * self.delta)
#        state_ = cp.Constant(state)
#
#        U = cp.Variable((self.N, self.N), nonneg=True)
#        ones = cp.Constant(np.ones((self.N,)))
#        constraints = [U @ ones == ones, ones @ U == ones]
#
#        lmbda = cp.Constant(self.gamma * np.ones(self.m))
#        group_error_ = cp.maximum(
#            [0.0] * self.M.shape[0], ((target_ - state_) / np.sum(self.M, axis=1))
#        )
#        item_error = self.M.T @ cp.multiply(lmbda, group_error_)
#        obj = (r.T @ U @ u) + (item_error @ U @ u)
#        prob = cp.Problem(cp.Maximize(obj), constraints)
#        prob.solve(cp.ECOS)
#        return self.sk.fit(U.value)
#
#
#class LPPController(BiostochasticController):
#    def __init__(self, T, N, M, delta, config):
#        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
#        self.gamma = config.get("gamma", 1)
#        self.C = config.get("C", [1] * M.shape[0])
#        super().__init__(T, N, M, delta, config)
#
#    def solve(self, state, r, tau):
#        M = cp.Constant(self.M)
#        v = cp.Constant(self.v)
#        r = cp.Constant(r)
#        u = cp.Constant(self.u)
#        U = cp.Variable((self.N, self.N), nonneg=True)
#        C = cp.Constant(self.C)
#
#        ## No intervention
#        # ranking_ = np.argsort(r.value)[::-1]
#        # obs_ = self.get_observation(ranking_)
#        # state_ = self.M.dot(obs_)
#
#        # No intervention
#        ones = cp.Constant(np.ones((self.N,)))
#        constraints = [U @ ones == ones, ones @ U == ones]
#
#        obj = r.T @ U @ u
#        prob = cp.Problem(cp.Maximize(obj), constraints)
#        prob.solve(cp.ECOS)
#        # assert np.isclose(M.value @ self.sk.fit(U.value) @ v.value, state_).all()
#
#        # Intervention
#        target_ = cp.Constant((tau + 1) / self.T * self.delta)
#        state_ = cp.Constant(state) + M.value @ self.sk.fit(U.value) @ v.value
#
#        U = cp.Variable((self.N, self.N), nonneg=True)
#        constraints = [U @ ones == ones, ones @ U == ones]
#
#        lmbda = self.gamma * C
#        group_error_ = cp.maximum(
#            [0.0] * self.M.shape[0], ((target_ - state_) / np.sum(self.M, axis=1))
#        )
#        item_error = self.M.T @ cp.multiply(lmbda, group_error_)
#        obj = (r.T @ U @ u) + (item_error.T @ U @ u)
#
#        # group_error_ = cp.maximum([0.] * self.M.shape[0], target_ - state_) / np.sum(self.M, axis=1)
#        # item_error = self.M.T @ group_error_
#        # lmbda = self.gamma * C
#        # obj = (r.T @ U @ u) + ((lmbda * item_error) @ U @ u)
#        # obj = (r.T + lmbda * cp.sum(group_error_)) @ U @ u
#        prob = cp.Problem(cp.Maximize(obj), constraints)
#        prob.solve(cp.ECOS)
#
#        ## Intervention
#        # s = r.value + lmbda * item_error.value
#        # ranking = np.argsort(s)[::-1]
#        return self.sk.fit(U.value)


class OracleController(BiostochasticController):
    def __init__(self, T, N, M, delta, config=None):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        super().__init__(T, N, M, delta, config)
        self.R = get_test_relevance()
        self.T_ = T
        logger.info("Learning optimal U for each query...")

        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))
        for _ in range(self.T_):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta)
        C = cp.Constant(self.C)

        utility = 0
        state = 0
        for i in range(self.T_):
            utility += R[i].T @ Us[i] @ u
            state += M @ Us[i] @ v
        group_error_ = cp.maximum(np.zeros(self.M.shape[0]), (target_ - state))

        obj = utility - C @ group_error_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        self.optimal_Us = [U.value for U in Us]

    def solve(self, state, r, tau):
        return self.sk.fit(self.optimal_Us[tau])


class CPController(BiostochasticController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        self.samples_ = None

        self.B = config.get("B", 100)
        self.B_online = config.get("B_online", self.B)
        assert self.B_online == 1

        self.relevance_type = config.get("relevance_type", None)
        self.hinge_min = config.get("hinge_min", 0.0)
        self.count = 0

        self.lr = config.get("bpc_lr", 0.01)
        self.init = config.get("bpc_init", "one")
        self.beta = config.get("beta", 0.5)
        self.eps = config.get("eps", 1e-5)
        self.C = np.array(config.get("C", np.ones(self.m)))
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)

        self.shuffle_bootstraps = config.get("shuffle_bootstraps", "true")

        if self.relevance_type == "offline_relevance":
            self.R = sample_relevance(self.train_size, replace=False)
        elif self.relevance_type == "online_relevance":
            self.R = sample_actual_relevance(None)
        elif self.relevance_type == "sequence_relevance":
            self.R = sample_sequence_relevance(None)
            assert self.shuffle_bootstraps == 'false'
            assert self.B <= self.R.shape[0]
            assert self.B_online <= self.R.shape[0]
        else:
            raise Exception("Unknown relevance type")
        self.NUM_OFFLINE = self.R.shape[0]

        assert self.NUM_OFFLINE > 0
        self.Us_offline = self.learn_offline()

        controller = config.get("controller")
        if self.init == "zero":
            self.lmbda = torch.zeros((self.B_online, self.m), requires_grad=True)
        elif self.init == "one":
            self.lmbda = torch.ones((self.B_online, self.m), requires_grad=True)
            with torch.no_grad():
                self.lmbda.data = torch.tensor(np.array(self.C)[None,:]).float()
        #self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr)
        self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr, betas=(self.beta, .999), eps=self.eps)

        logger.info("Init offline Us...")

    def bootstrap_online(self, tau):
        bs = np.arange(tau+1, len(self.Us_offline))
        return bs

    def learn_offline(self):
        def bootstrap_offline(index):
            if self.shuffle_bootstraps == "true":
                bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=False)
            elif self.relevance_type == "sequence_relevance":
                bs = np.arange(index*self.R.shape[1], (index+1)*self.R.shape[1])
            elif self.shuffle_bootstraps == "false":
                bs = np.arange(self.T_)
            return bs

        samples = [bootstrap_offline(index) for index in range(self.B)]
        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.T_):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = self.R.reshape(-1, self.R.shape[-1])
        R = cp.Constant(R)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)
        C = cp.Constant(self.C)

        utility = 0
        group_error_ = 0
        for i in range(self.B):
            state = 0
            for u_index, r_index in enumerate(samples[i]):
                state += M @ Us[u_index] @ v
                utility += R[r_index].T @ Us[u_index] @ u

            assert (u_index + 1) == self.T_
            group_error_ +=  C @ cp.maximum([0.0] * self.M.shape[0], (target_ - state))

        obj = utility - group_error_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return [U.value for U in Us]


    def act(self, state, r, tau):
        self.samples_ =  [self.bootstrap_online(tau) for _ in range(self.B_online)]
        U = self.solve(state.copy(), r, tau)
        self.solve_lmbda(state.copy(), tau, U)

        #self.Us_offline[tau] = U.copy()

        obs = U @ self.v
        exposure = self.M @ U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        #self.count += 1
        #self.realtime_delta = self.delta - state
        #self.realtime_T = self.T - self.count
        #print(f'state: {state} -- self.lmbda.data: {self.lmbda.data}')
        return state, util, obs, exposure

    def solve_lmbda(self, state, tau, U):
        samples = self.samples_.copy()
        target_ = self.delta_

        violation_ = []
        for i in range(self.B_online):
            state_ = state.copy()
            for u_index, r_index in enumerate(samples[i]):
                state_ += self.M @ self.Us_offline[r_index] @ self.v
            state_ += self.M @ U @ self.v
            violation_.append((target_ - state_))

        g = np.stack(violation_)

        # Gradient Ascent (-1)
        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(-1 * g).float()
        self.lmbda_opt.step()

        with torch.no_grad():
            min_ = torch.zeros(self.m)
            max_ = torch.tensor(self.C)
            self.lmbda.data = torch.clamp(self.lmbda.data, min=min_, max=max_).float()

class CP(CPController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        assert self.relevance_type in ["offline_relevance", 'sequence_relevance']
        assert self.B >= 1
        assert self.B_online == 1

    def solve(self, state, r, tau):
        samples = self.samples_.copy()

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        #R = cp.Constant(self.R)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)

        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u
        state_ = M @ U @ v
        assert self.B_online == 1

        obj = utility + lmbda[0] @ state_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)

class CPHinge(CPController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        assert self.relevance_type == "offline_relevance"
        assert self.B >= 1
        assert self.B_online == 1

    def solve(self, state, r, tau):
        samples = self.samples_.copy()

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        #R = cp.Constant(self.R)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u
        assert self.B_online == 1

        state_ = cp.Constant(state.copy())
        for u_index, r_index in enumerate(samples[0]):
            state_ += M @ cp.Constant(self.Us_offline[r_index]) @ v
        state_ += M @ U @ v

        obj = utility - lmbda[0] @ cp.maximum([self.hinge_min] * self.M.shape[0], (target_ - state_))
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)


class CPBController(BiostochasticController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        self.samples_ = None

        self.B = config.get("B", 100)
        self.B_online = config.get("B_online", self.B)

        self.relevance_type = config.get("relevance_type", None)
        self.hinge_min = config.get("hinge_min", 0.0)
        self.count = 0

        self.lr = config.get("bpc_lr", 0.01)
        self.init = config.get("bpc_init", "one")
        self.beta = config.get("beta", 0.5)
        self.eps = config.get("eps", 1e-5)
        self.C = np.array(config.get("C", np.ones(self.m)))
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)

        self.shuffle_bootstraps = config.get("shuffle_bootstraps", "true")

        if self.relevance_type == "offline_relevance":
            self.R = sample_relevance(self.T_, replace=False)
        elif self.relevance_type == "online_relevance":
            self.R = sample_actual_relevance(None)
        elif self.relevance_type == "sequence_relevance":
            self.R = sample_sequence_relevance(None)
            assert self.shuffle_bootstraps == 'false'
            assert self.B <= self.R.shape[0]
            assert self.B_online <= self.R.shape[0]
        else:
            raise Exception("Unknown relevance type")
        self.NUM_OFFLINE = self.R.shape[0]

        assert self.NUM_OFFLINE > 0
        self.queries = set()
        self.Us_offline = self.learn_offline()

        if self.init == "zero":
            self.lmbda = torch.zeros((self.B_online, self.m), requires_grad=True)
        elif self.init == "one":
            self.lmbda = torch.ones((self.B_online, self.m), requires_grad=True)
            with torch.no_grad():
                self.lmbda.data = torch.tensor(np.array(self.C)[None,:]).float()
        #self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr)
        self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr, betas=(self.beta, .999), eps=self.eps)

        logger.info("Init offline Us...")

    def bootstrap_online(self, tau):
        if self.shuffle_bootstraps == "true":
            bs = np.random.choice(list(self.queries), self.T_ - tau - 1, replace=False)
        else:
            bs = np.arange(tau+1, self.T_)
        return bs

    def learn_offline(self):
        def bootstrap_offline():
            if self.shuffle_bootstraps == "true":
                bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=False)
                self.queries |= set(list(bs))
            elif self.shuffle_bootstraps == "false":
                bs = np.arange(self.T_)
            return bs

        samples = [bootstrap_offline() for _ in range(self.B)]
        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.NUM_OFFLINE):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)
        C = cp.Constant(self.C)

        utility = 0
        group_error_ = 0
        for i in range(self.B):
            state = 0
            for _, r_index in enumerate(samples[i]):
                state += M @ Us[r_index] @ v
                utility += R[r_index].T @ Us[r_index] @ u

            group_error_ +=  C @ cp.maximum([0.0] * self.M.shape[0], (target_ - state))

        obj = utility - group_error_
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return [U.value for U in Us]


    def act(self, state, r, tau):
        self.samples_ =  [self.bootstrap_online(tau) for _ in range(self.B_online)]
        U = self.solve(state.copy(), r, tau)
        self.solve_lmbda(state.copy(), tau, U)

        #self.Us_offline[tau] = U.copy()

        obs = U @ self.v
        exposure = self.M @ U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        return state, util, obs, exposure

    def solve_lmbda(self, state, tau, U):
        samples = self.samples_.copy()
        target_ = self.delta_

        violation_ = []
        for i in range(self.B_online):
            state_ = state.copy()
            for _, r_index in enumerate(samples[i]):
                state_ += self.M @ self.Us_offline[r_index] @ self.v
            state_ += self.M @ U @ self.v
            violation_.append((target_ - state_))

        g = np.stack(violation_)

        # Gradient Ascent (-1)
        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(-1 * g).float()
        self.lmbda_opt.step()

        with torch.no_grad():
            min_ = torch.zeros(self.m)
            max_ = torch.tensor(self.C)
            self.lmbda.data = torch.clamp(self.lmbda.data, min=min_, max=max_).float()

class CPB(CPBController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        assert self.relevance_type == "offline_relevance"
        assert self.B >= 1

    def solve(self, state, r, tau):
        samples = self.samples_.copy()

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)

        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u
        state_ = M @ U @ v

        exposure = 0
        for i in range(self.B_online):
            exposure += lmbda[i] @ state_

        obj = utility + exposure / self.B_online
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)

class CPBHinge(CPBController):
    def __init__(self, T, N, M, delta, config=None):
        super().__init__(T, N, M, delta, config)
        assert self.relevance_type == "offline_relevance"
        assert self.B >= 1

    def solve(self, state, r, tau):
        samples = self.samples_.copy()

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        target_ = cp.Constant(self.delta_)

        # Clip lmabda
        lmbda = self.lmbda.detach().numpy()
        lmbda = cp.Constant(lmbda)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        utility = r.T @ U @ u

        exposure = 0
        for i in range(self.B_online):
            state_ = cp.Constant(state.copy())
            for _, r_index in enumerate(samples[0]):
                state_ += M @ cp.Constant(self.Us_offline[r_index]) @ v
            state_ += M @ U @ v
            exposure += lmbda[i] @ cp.maximum([self.hinge_min] * self.M.shape[0], (target_ - state_))

        obj = utility - exposure / self.B_online
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.MOSEK)
        return self.sk.fit(U.value)
