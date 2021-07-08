"""
Warning: The file contains bandit classes. The generic Bandit class is the only documented as all other classes are
child classes following the same pattern.
"""
import numpy as np
from tqdm import tqdm
import random
import scipy


class Bandit():
    """
    The Bandit class, is the most basic, generic parent class for bandits. All other bandit classes inherit at least
    from this class.
    """
    def __init__(self, env, init_pulls=1, horizon=int(1e3), normalisation=1, render=False, with_context=False,
                 seed=1234):
        """
        @param env: the bandit environment
        @type env: an OpenAI gym like environments, e.g. DssatEnv
        @param init_pulls: the number of times each arm will be pulled when initializing the bandit
        @type init_pulls: integer
        @param horizon: the bandit's horizon, T
        @type horizon: integer
        @param normalisation: a factor to normalize reward
        @type normalisation: 0 < float
        @param render: if the environment is rendered
        @type render: bool
        @param with_context: if the bandit is contextual
        @type with_context: bool
        @param seed: bandit's seed
        @type seed: integer
        """
        self.seed = seed
        self.env = env
        self.set_seed(seed)
        self.stop = False
        self.normalisation = normalisation
        self.render = render
        self.n_actions = self.env.action_space.n
        self.rewards = [[] for _ in range(self.n_actions)]
        self.pulls = np.zeros(self.n_actions)
        self.vars_ = np.repeat(np.nan, self.n_actions)
        self.means = np.repeat(np.nan, self.n_actions)
        self.criteria = np.repeat(np.nan, self.n_actions)
        self.t = 0
        self.frames = []
        self.action_history = []
        self.with_context = with_context
        if with_context:
            self.context = self.env.get_state()
        self.choice_criteria = np.amax
        self.init_pulls = init_pulls
        self.regrets = []
        self.horizon = horizon
        self.update_all = True
    # self.initialize() to call in child class !!

    def choice_function(self):
        """
        Selects the arm to play. In case of equality, returns the criteria with the minimum pulls.
        @return: the index of the arm to play
        @rtype: integer
        """
        criteria = np.array(self.criteria)
        all_candidates = np.argwhere(criteria == self.choice_criteria(criteria)).flatten()  # looking for multiple
        # min/max
        if len(all_candidates) == 1:
            selected, = all_candidates
        else:
            pulls = self.pulls[all_candidates]
            selected = all_candidates[np.argmin(pulls)]  # !! returns first min by default
        return selected

    def pull_once(self):
        """
        Pulls once all the arms for initialization
        @param seed: a seed
        @type seed: integer
        @return: nothing
        @rtype: None
        """
        for action_index in range(self.n_actions):
            self.action_history.append(action_index)
            self.pulls[action_index] += 1
            _, reward, _, _ = self.env.step(action_index)
            self.rewards[action_index].extend(np.array(reward) / self.normalisation)
            self.t += 1

    def initialize(self, init_pulls=None):
        """
        Initializes the bandit. !! Warning: has to be called in the child class only !!
        @param init_pulls: number of initial pulls
        @type init_pulls: integer
        @return: nothing
        @rtype: None
        """
        if init_pulls is None:
            init_pulls = self.init_pulls
        else:
            self.init_pulls = init_pulls
        for _ in range(init_pulls):  # to get vars correctly initialized
            self.pull_once()

        for action_index in range(self.n_actions):
            self.get_means(action_index)  # all has to be initialized for all actions at first
            self.get_vars(action_index)

    def step(self, action_index=None):  # action_index can be forced
        """

        @param action_index: action to play, if not None forces the action to be made, else,
         action making follows bandit policy.
        @type action_index: integer in range(self.env.n_actions)
        @return: nothing
        @rtype: None
        """
        if action_index is None:
            self.update_criteria()
            action_index = self.choice_function()
        self.action_history.append(action_index)
        context, reward, _, _ = self.env.step(action_index=action_index)
        self.rewards[action_index].extend(reward / self.normalisation)
        self.pulls[action_index] += 1
        self.t += 1
        if self.with_context:
            self.context = context

    def get_means(self, action_index):
        """
        Updates the mean of reward history for action of given index
        @param action_index: the index of the action to update the mean of rewards
        @type action_index: integer in range(self.env.n_actions)
        @return: nothing
        @rtype: None
        """
        self.means[action_index] = np.mean(self.rewards[action_index])

    def get_vars(self, action_index):
        """
        Updates the variance of reward history for action of given index
        @param action_index: the index of the action to update the variance of rewards
        @type action_index: integer in range(self.env.n_actions)
        @return: nothing
        @rtype: None
        """
        self.vars_[action_index] = np.var(self.rewards[action_index])

    def get_criteria(self, action_index):
        """
        Scores the arm of given index
        @param action_index: the index of the arm to score
        @type action_index: integer in range(self.env.n_actions)
        @return: nothing
        @rtype: None
        """
        pass  # to be defined in child class!!

    def update_criteria(self):
        """
        Updates the score of all arms if self.update_all is True (default), or
        only of the played arm else.
        @return: nothing
        @rtype: None
        """
        # if action_index provided, updates only for this action
        self.get_means(self.action_history[-1])  # only needed for last made action
        self.get_vars(self.action_history[-1])
        if self.update_all:
            for action_index in range(self.n_actions):  # updates are needed for all actions
                self.get_criteria(action_index=action_index)
        else:
            self.get_criteria(action_index=self.action_history[-1])

    def set_seed(self, seed=None):
        """
        Sets the seed of the bandit. In particular for bandits with a stochastic policy. Propagates the seed to
        the environment
        @param seed: the seed to set
        @type seed: integer
        @return: nothing
        @rtype: None
        """
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        try:
            self.env.set_seed(seed)
        except:
            print('Warning: env.set_seed(seed) failed')

    def reset(self):
        """
        Resets all relevant attributes before a new run.
        @return: nothing
        @rtype: None
        """
        self.rewards = [[] for _ in range(self.n_actions)]
        self.pulls = np.zeros(self.n_actions)
        self.vars_ = np.repeat(np.nan, self.n_actions)
        self.means = np.repeat(np.nan, self.n_actions)
        self.criteria = np.repeat(np.nan, self.n_actions)
        self.t = 0
        self.frames = []
        self.context = None
        self.action_history = []
        self.regrets = []
        self.extra_reset()
        self.initialize()

    def extra_reset(self):
        """
        In case extra attributes need a reset, this function provides flexibility.
        @return: nothing
        @rtype: None
        """
        pass

    def get_regret(self):
        """
        Returns the cumulated regret of the bandit. Called after self.t=horizon.
        @return: regret
        @rtype: list of shape (horizon)
        """
        best_value = self.choice_criteria(self.env.true_params)
        best_value_cumsum = best_value * np.arange(1, self.horizon + 1)
        action_value_cumsum = np.cumsum(np.array(self.env.true_params)[self.action_history])
        if self.choice_criteria == np.amax:
            regrets = best_value_cumsum - action_value_cumsum
        else:
            regrets = action_value_cumsum - best_value_cumsum
        self.regrets.extend(regrets)
        return regrets

    def loop(self, horizon=None):
        """
        Makes the bandit step self.horizon times in total if horizon is None, else horizon times in total.
        @return: nothing
        @rtype: None
        """
        if horizon is None:
            horizon = self.horizon
        for _ in tqdm(range(horizon - self.init_pulls * self.n_actions)):
            self.step()

class Uniform(Bandit):
    """
    A bandit with a uniform random policy.
    """
    def __init__(self, env, init_pulls=1, horizon=None, **kwargs):
        super().__init__(env, init_pulls=init_pulls, horizon=horizon)
        self.initialize()

    def get_criteria(self, action_index):
        criteria = np.random.uniform(0, 1)
        self.criteria[action_index] = criteria
        return criteria

    def get_means(self, action_index):
        pass

    def get_vars(self, action_index):
        pass


class Determinist(Uniform):
    """
    A bandit with a determinist policy. Cycles trough list if available actions.
    """
    def __init__(self, env, init_pulls=1, horizon=None, **kwargs):
        super().__init__(env, init_pulls=init_pulls, horizon=horizon)
        self.initialize()

    def get_criteria(self, action_index):
        criteria = 0
        if action_index == (self.t // self.n_actions + self.t % self.n_actions):
            criteria = 1
        self.criteria[action_index] = criteria
        return criteria


if __name__ == '__main__':
    pass
