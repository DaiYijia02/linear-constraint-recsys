import random

import torch

import cvxpy as cp
import numpy as np
from loguru import logger
from birkhoff import TOLERANCE
from sinkhorn_knopp import sinkhorn_knopp as skp
from simulator import Simulator
from config import dataConfig as dc
from collections import defaultdict
from utils import (
    bvn_decomp,
    dcg_util_fn,
    exposure_fn,
    sample_relevance,
    ewma
)
import multiprocessing as mp
from functools import partial
import pdb
import sys

class BaseController(object):
    # T, N, M, delta, config
    def __init__(self, T, N, M, delta, config):
        self.T, self.N = T, N
        self.M = M
        self.m = M.shape[0]
        self.delta = delta
        self.u = dcg_util_fn(np.arange(N) + 1)
        self.v = exposure_fn(np.arange(N) + 1)
        self.C = config.get("C", [1.] * M.shape[0])
        #self.v = dcg_util_fn(np.arange(N) + 1)

    def get_observation(self, ranking):
        rank = np.zeros(self.N, dtype=np.int32)
        rank[ranking] = np.arange(self.N)
        return self.v[rank]

    def get_utility(self, ranking, r):
        return self.u.dot(r[ranking])

    # (state, r, tau)
    def get_action(self, state, r, tau):
        #raise NotImplementedError
        ranking_ = np.argsort(r)[::-1]
        return ranking_, 'optimal'

    def act(self, state, r, tau):
        ranking,  = self.get_action(state, r, tau)
        obs = self.get_observation(ranking)
        util = self.get_utility(ranking, r)
        state += self.M.dot(obs)
        return state, util, obs

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

        group_error = np.clip(setpoint - state_, 0, None) / \
            np.sum(self.M, axis=1)
        item_error = self.M.T.dot(group_error)

        s = r + (self.lmbda * self.C) * item_error
        ranking = np.argsort(s)[::-1]
        return ranking, 'optimal'


class PController(BasePController):
    def __init__(self, T, N, M, delta, config):
        #if "lambda" in config:
        logger.debug(f"Using lambda={config['lambda']}")
        super().__init__(T, N, M, delta, config, config["lambda"])
        #else:
        #    self.delta = delta
        #    R = sample_relevance(T)
        #    best_objective = -np.inf
        #    best_mean = None
        #    best_lmbda = None
        #    best_unsat_mean = None
        #    best_unsat_std = None
        #    C = config.get("C", 1)
        #    m = M.shape[0]
        #    #ls = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300, 1000]
        #    ls = [100, 10, 1, .1, .01, .001]
        #    for lmbda in ls:
        #        controller = BasePController(T, N, M, delta, config, lmbda )
        #        offline_simulator = Simulator(controller, m, dc.N, dc.MAX_UTIL, self.delta, C, test=False)
        #        state, utility, obs = offline_simulator.simulate(R)
        #        metrics = offline_simulator.get_metrics(self.delta)

        #        exp_ratio = np.clip(delta - state, 0, None) / delta
        #        weighted_objective = metrics['Weighted Objective']
        #        if best_objective < weighted_objective:
        #        # if mean_ratio < best_mean and\
        #        #    metrics['Unsatisfaction Mean'] == 0 and\
        #        #    metrics['Unsatisfaction Std'] == 0:
        #            best_objective = weighted_objective
        #            best_lmbda = lmbda
        #            best_unsat_mean = metrics['Unsatisfaction Mean']
        #            best_unsat_std = metrics['Unsatisfaction Std']
        #    logger.debug(f"Best lambda: {best_lmbda}")
        #    print(f"Best lambda: {best_lmbda} best_objective: {best_objective} best_unsat_std: {best_unsat_std} best_unsat_mean: {best_unsat_mean}")
        #    super().__init__(T, N, M, delta, config, best_lmbda)


class BiostochasticController(BaseController):
    def solve(self):
        pass

    # TODO: remove the below function and enable the next function
    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)
        return state, util, obs

    def get_action(self, state, r):
        U  = self.solve(state, r)
        permutations, coefficients = bvn_decomp(U)
        permutation = random.choices(permutations, weights=coefficients)[0]
        _, ranking = np.nonzero(permutation.T)
        return ranking

class BPController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        target_ = cp.Constant((tau+1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        lmbda = self.gamma * C
        group_error_ = cp.maximum([0.] * self.M.shape[0], (target_ - state_))
        obj = (r.T @ U @ u) - lmbda @ group_error_

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class BPApproxController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        target_ = cp.Constant((tau+1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        lmbda = self.gamma * C
        group_error_ =  ((target_ - state_) / np.sum(self.M, axis=1))
        item_error = cp.maximum([0.] * self.M.shape[1], group_error_)
        obj = (r.T @ U @ u) - (self.M.T @ cp.multiply(lmbda,item_error) @ u)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class BPSoftController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        kappa = cp.Variable(self.M.shape[0])

        target_ = cp.Constant((tau+2) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        lmbda = self.gamma * self.C
        obj = (r.T @ U @ u) - cp.sum(lmbda * kappa)  #cp.sum(kappa)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones,
                       ones @ U == ones,
                       kappa >= target_ - state_,
                       kappa >= [0.,0.]]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class BPSoftApproxController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        kappa = cp.Variable(self.M.shape[1])

        target_ = cp.Constant((tau+1) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        lmbda = self.gamma * self.C
        obj = (r.T @ U @ u) - (kappa @ u)  #cp.sum(kappa)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones,
                       ones @ U == ones,
                       kappa >= lmbda * self.M.T @ ((target_-state_) / np.sum(self.M, axis=1)),
                       kappa >= [0.] * self.M.shape[1]]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class LPPNoStateController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1.] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        # Intervention
        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state)

        U = cp.Variable((self.N, self.N), nonneg=True)
        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        lmbda = self.gamma * C
        group_error_ = cp.maximum([0.] * self.M.shape[0], ((target_ - state_) / np.sum(self.M, axis=1)) )
        item_error = self.M.T @ cp.multiply(lmbda, group_error_)
        obj = (r.T @ U @ u) + (item_error @ U @ u)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class LPPNoStateApproxController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)
        C = cp.Constant(self.C)

        # Intervention
        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state)

        U = cp.Variable((self.N, self.N), nonneg=True)
        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        lmbda = self.gamma * C
        group_error_ = ((target_ - state_) / np.sum(self.M, axis=1))
        item_error = cp.maximum([0.] * self.M.shape[1], self.M.T @ cp.multiply(lmbda, group_error_))
        obj = (r.T @ U @ u) + (item_error @ U @ u)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)



class LPPController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)

        ## No intervention
        #ranking_ = np.argsort(r.value)[::-1]
        #obs_ = self.get_observation(ranking_)
        #state_ = self.M.dot(obs_)

        # No intervention
        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones, ones @ U == ones]

        obj = (r.T @ U @ u)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        #assert np.isclose(M.value @ self.sk.fit(U.value) @ v.value, state_).all()

        # Intervention
        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state) + M.value @ self.sk.fit(U.value) @ v.value

        U = cp.Variable((self.N, self.N), nonneg=True)
        constraints = [U @ ones == ones, ones @ U == ones]

        group_error_ = cp.maximum([0.] * self.M.shape[0], target_ - state_) / np.sum(self.M, axis=1)
        item_error = self.M.T @ group_error_
        lmbda = self.gamma * self.C
        obj = (r.T @ U @ u) + ((lmbda * item_error) @ U @ u)
        #obj = (r.T + lmbda * cp.sum(group_error_)) @ U @ u
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)

        ## Intervention
        #s = r.value + lmbda * item_error.value
        #ranking = np.argsort(s)[::-1]
        return self.sk.fit(U.value)


class LPSoftController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U_hat = cp.Variable((self.N, self.N), nonneg=True)

        ## No intervention
        #ranking_ = np.argsort(r.value)[::-1]
        #obs_ = self.get_observation(ranking_)
        #state_ = self.M.dot(obs_)

        # No intervention
        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U_hat @ ones == ones, ones @ U_hat == ones]

        obj = (r.T @ U_hat @ u)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        #assert np.isclose(M.value @ self.sk.fit(U.value) @ v.value, state_).all()

        # Intervention
        kappa = cp.Variable(self.M.shape[0])
        U = cp.Variable((self.N, self.N), nonneg=True)
        target_ = cp.Constant((tau + 1) / self.T * self.delta)
        state_ = cp.Constant(state) + M.value @ self.sk.fit(U_hat.value) @ v.value
        lmbda = self.gamma * self.C
        #obj = (r.T @ U @ u) + lmbda * cp.sum((self.M.T @ kappa) @ U @ u)
        obj = (r.T + lmbda * (self.M.T @ kappa)) @ U @ u

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones,
                       ones @ U == ones,
                       kappa >= target_ - state_,
                       kappa >= [0., 0.]]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class OnlineLinearPrograming(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        super().__init__(T, N, M, delta, config)
        self.lr = config.get('olp_lr', 0.01)
        self.init = config.get('olp_init', 'one')
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        if self.init == 'zero':
            self.lmbda = torch.zeros((self.m,), requires_grad=True)
        elif self.init == 'one':
            self.lmbda = torch.ones((self.m,), requires_grad=True)
        self.lmbda_opt = torch.optim.Adam([self.lmbda], lr=self.lr)
        self.delta_ = delta
        self.realtime_delta = delta

        self.Z = 1.

        self.lp = OLPParameterLP(np.zeros(self.m), np.zeros(self.N), self.Z, T, N, M, self.v, self.u, self.delta_, self.lmbda.detach().numpy())

    def act(self, state, r, tau):
        U = self.lp.solve_U(state, r, tau, self.Z, self.delta_, self.lmbda.detach().numpy())

        obs = U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)

        self.realtime_delta = self.delta - state
        self.solve_lmbda(self.M.dot(obs), tau)
        return state, util, obs

    def solve_lmbda(self, state, tau):
        setpoint = (1/self.T) * self.delta_
        g = (setpoint - state)

        self.lmbda.sum().backward()
        self.lmbda.grad = torch.tensor(g).float()
        self.lmbda_opt.step()
        self.lmbda_opt.param_groups[0]['lr'] = self.lr / np.sqrt(tau + 1)

    def get_action(self, state, r):
         raise Exception('Not Implemented')

class OLPParameterLP:
    def __init__(self, state, r, Z, T, N, M, v, u, delta_, lmbda):
        self.U = cp.Variable((N, N), nonneg=True)

        self.M = cp.Constant(M)
        self.v = cp.Constant(v)
        self.u = cp.Constant(u)

        self.r = cp.Parameter(r.shape[0])
        self.delta = cp.Parameter(delta_.shape)
        self.Z = cp.Constant(Z)
        self.lmbda = cp.Parameter(lmbda.shape)

        self.ones = cp.Constant(np.ones((N,)))
        self.constraints = [
            self.U @ self.ones == self.ones,
            self.ones @ self.U == self.ones,
        ]

        self.utility = cp.sum(cp.multiply(self.r, self.U @ self.u)) - self.Z * cp.sum(cp.multiply(self.lmbda, self.M @ self.U @ self.v))
        self.prob = cp.Problem(cp.Maximize(self.utility), self.constraints)
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)

    def solve_U(self, state, r, tau, Z, delta_, lmbda):
        self.r.value = r
        self.delta.value = delta_
        self.lmbda.value = lmbda
        self.prob.solve(cp.ECOS, warm_start=True)
        return self.sk.fit(self.U.value)

class MPController(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.H = config.get("H", 1)
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        sample = sample_relevance(self.H - 1)

        Us = []
        for _ in range(self.H):
            U = cp.Variable((self.N, self.N), nonneg=True)
            Us.append(U)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(np.vstack((r, sample)))
        u = cp.Constant(self.u)

        obj = 0
        for i in range(self.H):
            obj += cp.sum(cp.multiply(R[i], Us[i] @ u))

        target = (self.H + tau) / self.T * self.delta
        z = cp.Constant(target - state)
        for i in range(self.H):
            z -= M @ Us[i] @ v

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            z <= 0,
        ]

        for i in range(self.H):
            constraints.append(Us[i] @ ones == ones)
            constraints.append(ones @ Us[i] == ones)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(Us[0].value)


class SMPController(BiostochasticController):
    def __init__(self, T, N, M, delta, config=None):
        self.B = config.get("B", 100)
        self.B_online = config.get("B_online", self.B)
        self.R = None
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)

        self.relearn_iterations = config.get("relearn_iterations", None)
        self.train_size = config.get("train_size", None)
        self.gamma = config.get("gamma", 1)

        self.realtime_delta = delta
        self.realtime_T = T
        self.delta_ = delta
        self.T_ = T
        super().__init__(T, N, M, delta, config)

        # History
        self.count = 0
        self._Us = []
        self._r = []
        # Offline Collection
        self.R = sample_relevance(self.train_size, replace=False)
        self.NUM_OFFLINE = self.R.shape[0]

        if self.NUM_OFFLINE > 0:
            self.Us_offline = self.learn_offline()
        else:
            self.Us_offline = []
        logger.info("Init offline Us...")

    def act(self, state, r, tau):
        U = self.solve(state, r, tau)

        obs = U @ self.v
        util = r.T @ U @ self.u
        state += self.M.dot(obs)
        self._Us.append(U)
        self._r.append(r)
        self.count += 1
        self.realtime_delta = self.delta - state
        self.realtime_T = self.T - self.count

        if (
            self.relearn_iterations
            and self.count % self.relearn_iterations == 0
            and self.realtime_T > 0
        ):
            self.delta_ = self.realtime_delta
            self.T_ = self.realtime_T
            self.Us_offline = self.learn_offline()

        ##self.Us_offline[tau] = U
        return state, util, obs

    def learn_offline(self):
        def bootstrap_offline():
            bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=True)
            b = np.zeros((self.NUM_OFFLINE,), dtype=np.int)
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_offline() for _ in range(self.B)]

        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.NUM_OFFLINE):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        violations = []
        sum_violations = 0
        for _ in range(self.B):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        delta = cp.Constant(self.delta_)

        utility = 0
        for i in range(self.NUM_OFFLINE):
            utility += R[i].T @ Us[i] @ u

        for i in range(self.B):
            exposure = 0
            for j in range(self.NUM_OFFLINE):
                exposure += M @ Us[j] @ v * samples[i][j]
            constraints.append(exposure >= delta - violations[i])

        obj = self.T_ / self.NUM_OFFLINE * utility - (self.C * self.gamma) / self.B * cp.sum(
            sum_violations
        )
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return [U.value for U in Us]

    def solve(self, state, r, tau):
        def bootstrap_online():
            bs = np.random.choice(self.NUM_OFFLINE + 1,
                                  self.T_ - 1, replace=True)
            b = np.zeros((self.NUM_OFFLINE + 1,), dtype=np.int)
            b[-1] = 1
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_online() for _ in range(self.B_online)]

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        u = cp.Constant(self.u)
        r = cp.Constant(r)
        delta = cp.Constant(self.delta_)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        violations = []
        sum_violations = 0
        for _ in range(self.B_online):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        for i in range(self.B_online):
            exposure = 0
            for j in range(self.NUM_OFFLINE):
                exposure += M @ self.Us_offline[j] @ v * samples[i][j]
            exposure += M @ U @ v * samples[i][-1]
            constraints.append(exposure >= delta - violations[i])

        utility = r.T @ U @ u
        obj = (self.NUM_OFFLINE + self.T_) / (
            self.NUM_OFFLINE + 1
        ) * utility - (self.C * self.gamma) / self.B_online * cp.sum(sum_violations)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)


class SMPCA(SMPController):

    def solve(self, state, r, tau):
        def bootstrap_online():
            start_index = tau if dc.IS_TEMPORAL else 0
            bs = np.random.choice(np.arange(start_index, self.NUM_OFFLINE),
                                  self.realtime_T - 1, replace=True)
            b = np.zeros((self.NUM_OFFLINE + 1,), dtype=np.int)
            b[-1] = 1
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_online() for _ in range(self.B_online)]

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        u = cp.Constant(self.u)
        r = cp.Constant(r)
        delta = cp.Constant(self.realtime_delta)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        violations = []
        sum_violations = 0
        for _ in range(self.B_online):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        for i in range(self.B_online):
            exposure = 0
            for j in range(self.NUM_OFFLINE):
                exposure += M @ self.Us_offline[j] @ v * samples[i][j]
            exposure += M @ U @ v * samples[i][-1]
            constraints.append(exposure >= delta - violations[i])

        utility = r.T @ U @ u
        obj = (self.NUM_OFFLINE + self.realtime_T) / (
            self.NUM_OFFLINE + 1
        ) * utility - (self.C * self.gamma) / self.B_online * cp.sum(sum_violations)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class SMPCB(SMPController):

    def learn_offline(self):
        def bootstrap_offline():
            bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=True)
            b = np.zeros((self.NUM_OFFLINE,), dtype=np.int)
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_offline() for _ in range(self.B)]

        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.NUM_OFFLINE):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        violations = []
        sum_violations = 0
        for _ in range(self.B):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        delta = cp.Constant(self.delta_)

        #utility = 0
        #for i in range(self.NUM_OFFLINE):
        #    utility += cp.sum(cp.multiply(R[i], Us[i] @ u))

        utility = 0
        for i in range(self.B):
            exposure = 0
            for j in range(self.NUM_OFFLINE):
                exposure += M @ Us[j] @ v * samples[i][j]
                utility += cp.sum(cp.multiply(R[j], Us[j] @ u) * samples[i][j])
            constraints.append(exposure >= delta - violations[i])

        obj = (1. / self.B) * utility - ((self.C * self.gamma) / self.B) * cp.sum(sum_violations)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return [U.value for U in Us]

    def solve(self, state, r, tau):
        def bootstrap_online():
            start_index = 0 #tau if dc.IS_TEMPORAL else 0
            bs = np.random.choice(np.arange(start_index, self.NUM_OFFLINE),
                                  self.realtime_T - 1, replace=True)
            b = np.zeros((self.NUM_OFFLINE + 1,), dtype=np.int)
            b[-1] = 1
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_online() for _ in range(self.B_online)]

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        r = cp.Constant(r)
        delta = cp.Constant(self.realtime_delta)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        violations = []
        sum_violations = 0
        for _ in range(self.B_online):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        utility = 0
        for i in range(self.B_online):
            exposure = 0
            for j in range(self.NUM_OFFLINE):
                exposure += M @ self.Us_offline[j] @ v * samples[i][j]
                utility += ((R[j].T @ self.Us_offline[j] @ u) * samples[i][j])
            exposure += M @ U @ v * samples[i][-1]
            utility += ((r.T @ U @ u) * samples[i][-1])
            constraints.append(exposure >= delta - violations[i])

        obj = (1. / self.B_online) * utility - ((self.C * self.gamma) / self.B_online) * cp.sum(sum_violations)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class SMPCC(SMPController):

    def learn_offline(self):
        def bootstrap_offline():
            bs = np.random.choice(self.NUM_OFFLINE, self.T_, replace=True)
            b = np.zeros((self.NUM_OFFLINE,), dtype=np.int)
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_offline() for _ in range(self.B)]

        Us = []
        constraints = []
        ones = cp.Constant(np.ones((self.N,)))

        for _ in range(self.T_):
            U = cp.Variable((self.N, self.N), nonneg=True)
            constraints.append(U @ ones == ones)
            constraints.append(ones @ U == ones)
            Us.append(U)

        violations = []
        sum_violations = 0
        for _ in range(self.B):
            xi = cp.Variable((self.m,), nonneg=True)
            violations.append(xi)
            sum_violations += xi

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        delta = cp.Constant(self.delta_)

        utility = 0
        for i in range(self.B):
            exposure = 0
            u_index = 0
            for j in range(self.NUM_OFFLINE):
                if samples[i][j] > 0:
                    for _ in range(samples[i][j]):
                        exposure += M @ Us[u_index] @ v
                        utility += R[j].T @ Us[u_index] @ u
                        u_index += 1
            assert (u_index == self.T_)
            constraints.append(exposure >= delta - violations[i])

        obj = (1. / self.B) * utility - ((self.C * self.gamma) / self.B) * cp.sum(sum_violations)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return [U.value for U in Us]
        #return [np.eye(self.M.shape[1]) for _ in range(self.T_)]

    def solve(self, state, r, tau):
        def bootstrap_online():
            start_index = 0 #tau if dc.IS_TEMPORAL else 0
            if self.NUM_OFFLINE > 0:
                bs = np.random.choice(np.arange(start_index, self.NUM_OFFLINE),
                                      self.realtime_T - 1, replace=True)
            else:
                bs = []

            b = np.zeros((self.NUM_OFFLINE + 1,), dtype=np.int)
            b[-1] = 1
            for i in bs:
                b[i] += 1
            return b

        samples = [bootstrap_online() for _ in range(self.B_online)]

        U = cp.Variable((self.N, self.N), nonneg=True)

        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        if self.NUM_OFFLINE > 0:
            R = cp.Constant(self.R)
        u = cp.Constant(self.u)
        r = cp.Constant(r)
        #delta = cp.Constant(self.realtime_delta)
        target_ = cp.Constant((tau+2) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [
            U @ ones == ones,
            ones @ U == ones,
        ]

        #violations = []
        #sum_violations = 0
        #for _ in range(self.B_online):
        #    xi = cp.Variable((self.m,), nonneg=True)
        #    violations.append(xi)
        #    sum_violations += xi

        #utility = 0
        #for i in range(self.B_online):
        #    exposure = 0
        #    u_index = tau + 1
        #    for j in range(self.NUM_OFFLINE):
        #        if samples[i][j] > 0:
        #            for _ in range(samples[i][j]):
        #                exposure += M @ self.Us_offline[u_index] @ v
        #                utility += R[j].T @ self.Us_offline[u_index] @ u
        #                u_index += 1
        #    #assert(u_index - (tau + 1) + samples[i][-1] == self.realtime_T )
        #    exposure += (cp.Constant(state) + M @ U @ v)
        #    #constraints.append(exposure >= delta - violations[i])
        constraints.append(self.C * (target_ - state_) <= [0., 0.])

        utility = (r.T @ U @ u)

        #obj = (1. / self.B_online) * utility
        obj = utility
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)

class SMPCD(BiostochasticController):
    def __init__(self, T, N, M, delta, config):
        self.sk = skp.SinkhornKnopp(epsilon=TOLERANCE)
        self.gamma = config.get("gamma", 1)
        self.C = config.get("C", [1] * M.shape[0])
        super().__init__(T, N, M, delta, config)

    def solve(self, state, r, tau):
        M = cp.Constant(self.M)
        v = cp.Constant(self.v)
        r = cp.Constant(r)
        u = cp.Constant(self.u)
        U = cp.Variable((self.N, self.N), nonneg=True)

        target_ = cp.Constant((tau+2) / self.T * self.delta)
        state_ = cp.Constant(state) + M @ U @ v
        lmbda = self.gamma * self.C
        obj = (r.T @ U @ u)

        ones = cp.Constant(np.ones((self.N,)))
        constraints = [U @ ones == ones,
                       ones @ U == ones,
                       lmbda * (target_ - state_) <= [0,0]]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(cp.ECOS)
        return self.sk.fit(U.value)



