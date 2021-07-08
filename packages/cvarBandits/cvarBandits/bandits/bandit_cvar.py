import sys, pdb
import numpy as np
from ..bandits import bandit_base
import scipy.stats as st
from ..utils import estimators
import random
import scipy

class CVAR(bandit_base.Bandit):
    def __init__(self, env, delta=None, alpha=None, limits=None, init_pulls=1, horizon=None, seed=None):
        super().__init__(env, init_pulls=init_pulls, horizon=horizon, seed=seed)
        if delta is None:
            delta = env.delta
        if alpha is None:
            alpha = env.alpha
        if limits is None:
            limits = env.supp
        self.delta = delta
        self.alpha = alpha
        self.limits = limits

class CVAR_UCB(CVAR):
    def __init__(self, env, bt_choice='union', delta=None, alpha=None, limits=None, init_pulls=1, horizon=None,
                 time_uniform_peeling=False, seed=None):
        super().__init__(env, delta=delta, alpha=alpha, limits=limits, init_pulls=init_pulls, horizon=horizon,
                         seed=seed)
        self.bt_choice = bt_choice
        self.time_uniform_peeling = time_uniform_peeling
        self.initialize()

    def get_criteria(self, action_index):
        samples = self.rewards[action_index]
        n = len(samples)
        sorted_samples = sorted(samples)
        b_t = np.sqrt(np.log(2 * self.horizon ** 2) / (2 * n))
        term1 = np.diff(sorted_samples, append=self.env.supp[1])
        term2 = np.maximum(np.minimum([i / n - b_t for i in range(1, n + 1)], self.alpha), 0)
        term = term1 * term2
        criteria = self.env.supp[1] - 1 / self.alpha * term.sum()
        self.criteria[action_index] = criteria
        return criteria

    def get_means(self, action_index):
        pass

    def get_vars(self, action_index):
        pass

class BROWN_UCB(CVAR):

    def __init__(self, env, alpha=None, delta=None, init_pulls=1, limits=None, horizon=None, seed=None):
        super().__init__(env, delta=delta, alpha=alpha, limits=limits, init_pulls=init_pulls, horizon=horizon,
                         seed=seed)
        self.initialize()

    def get_criteria(self, action_index):
        samples = self.rewards[action_index]
        n = len(samples)
        bonus = self.env.supp[-1] / self.alpha * np.sqrt(np.log(self.t) / n)
        criteria = estimators.moment_empirical_cvar(samples, alpha=self.alpha) + bonus
        self.criteria[action_index] = criteria
        return criteria

class B_CVTS(CVAR):
    def __init__(self, env, alpha=None, delta=None, init_pulls=1, limits=None, horizon=None, seed=None):
        super().__init__(env, delta=delta, alpha=alpha, limits=limits, init_pulls=init_pulls, horizon=horizon,
                         seed=seed)
        self.rewards_ = [[] for _ in range(self.n_actions)]
        self.initialize()

    def initialize(self, init_pulls=None):
        for action_index in range(self.n_actions):
            self.rewards_[action_index].append(self.env.supp[-1])  # reward with supp at beginning
            _, reward, _, _ = self.env.step(action_index)
            self.rewards[action_index].extend(np.array(reward) / self.normalisation)
            self.pulls[action_index] += 1
            self.t += 1
            self.action_history.append(action_index)

    def get_criteria(self, action_index):
        samples = np.sort(self.rewards_[action_index])
        weights = np.random.dirichlet(np.ones(len(samples)))
        q_index = int(np.sum(weights.cumsum() <= self.alpha))
        q_value = samples[q_index]
        criteria = q_value - 1 / self.alpha * (weights * np.maximum(q_value - samples, 0)).sum()
        self.criteria[action_index] = criteria
        return criteria

    def step(self, action_index=None):  # action_index can be forced
        if action_index is None:
            self.update_criteria()
            action_index = self.choice_function()
        self.action_history.append(action_index)
        context, reward, _, _ = self.env.step(action_index=action_index)
        self.rewards[action_index].extend(reward / self.normalisation)
        self.rewards_[action_index].extend(reward / self.normalisation)
        self.pulls[action_index] += 1
        self.t += 1
        if self.with_context:
            self.context = context

    def extra_reset(self):
        self.rewards_ = [[] for _ in range(self.n_actions)]

if __name__ == '__main__':
    pass