"""
OpenAI gym (https://gym.openai.com/) based DSSAT (https://dssat.net) crop-model environment for bandits.
"""
import matplotlib
import matplotlib.ticker as mtick

matplotlib.use('Agg')
# matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['pdf.fonttype'] = 42
import gym
import numpy as np
from dssatUtils.dssatIntegration import dssatIntegration
from cvarBandits.utils.estimators import moment_empirical_cvar
import pickle
import itertools
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme()
sns.set_context("paper")
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

import datetime
import os, sys
import numpy as np
import random
import pdb
import pprint
from itertools import cycle

import logging


class DssatEnv(gym.Env):
    """
    The DssatEnv is a python wrapper between OpenAI Gym and the DSSAT python class using the fortran based DSSAT
    crop-simulator (https://dssat.net/). It is an easy to manipulate bandit-geared environment. DSSAT parametrization
    is provided to the DSSAT python class trough the dssat_param_dic parameter. Actions can be made on planting date
    choice and/or cultivar choice. If action is made on cultivar choice, an additional csv file has to be provided and
    indicated in python DSSAT class with the parameter cultivar_path. This file define possible random choices
    of cultivars. An example is given with cultivars.csv. A random soil option can be activated in python DSSAT class.
    If so, a relevant csv has to provided to python DSSAT class with the parameter soil_path. This file define possible
    random choices of soils. An example is given with soils.csv. A random weather option can be activated in python
    DSSAT class with the parameter random_weather. If so, DSSAT's WGEN internal weather generator is used ; the user
    has to insure that relevant WGEN input_files are available to DSSAT for the given experiment. If some random parameters
    are activated, a random value is sorted each time the step functions is called. Before deleting DssatEnv, you should
    delete DSSAT integration temporary folder by calling self.dssat.close().
    """

    def __init__(self):
        pass

    def _init_(self, dssat_param_dic, env_param_dic, alpha=.1, delta=1e-3, n_samples=int(1e6),
               eta_max=15000, normalize=False, date_delta=20, date_steps=4, sowing_date=True, cultivar=False,
               is_cvar=False, stateless=True, seed=1234, *args, **kwargs):
        """
        Initialization to be made after gym environment initialization (at the time gym does not allow nested init.).
        @param dssat_param_dic: parameters to be sent to DSSAT class
        @type dssat_param_dic: dictionary
        @param env_param_dic: parameters to be sent to DssatEnv class
        @type env_param_dic: dictionary
        @param alpha: value of CVaR's alpha
        @type alpha: float
        @param delta: delta value for confidence bound estimations
        @type delta: 0 <= float <= 1
        @param n_samples: number of samples to estimate DssatEnv arms
        @type n_samples: integer
        @param eta_max: yield upper bound, kg/ha
        @type eta_max: float
        @param normalize: if yield observation to be normalized by eta_max
        @type normalize: bool
        @param date_delta: the time step in days between planting dates to be considered
        @type date_delta: integer
        @param date_steps: the number of planting dates to be considered from the initial planting date
        @type date_steps: integer
        @param sowing_date: if action making includes choosing a planting date
        @type sowing_date: bool
        @param cultivar: if action making includes choosing a cultivar
        @type cultivar: bool
        @param is_cvar: if cvar is the criteria to be maximized (if not, then the mean is considered)
        @type is_cvar: bool
        @param is_cvar: if the step function has to return contextual information
        @type is_cvar: bool
        @param seed: seed for the DssatEnv
        @type seed: int
        @return: nothing
        @rtype: None
        """
        super().__init__()
        self.seed = seed
        self.is_cvar = is_cvar
        self.alpha = alpha
        self.delta = delta
        self.n_samples = n_samples
        self.normalize = normalize
        self.eta_max = eta_max
        self.date_delta = date_delta
        self.date_steps = date_steps
        self.sowing_date = sowing_date
        self.cultivar = cultivar
        self.stateless = stateless
        self.__dict__.update(env_param_dic)  # overwrites default values if provided
        dssat_param_dic['cultivar_bool'] = self.cultivar
        dssat_param_dic['stateless'] = self.stateless
        self.env_param_dic = env_param_dic
        self.env_param_dic_loaded_samples = None
        self.dssat_param_dic_loaded_samples = None
        if self.normalize:
            self.supp = [0, 1]
        else:
            self.supp = [0, eta_max]
        self.samples = None
        self.restriction_list = None
        self.dssat_param_dic = dssat_param_dic
        self.check_for_dssat_auxiliary_files()
        self.dssat = dssatIntegration.DSSAT(**self.dssat_param_dic)
        self.dssat.seed = self.seed
        self.sample_loading_path = None
        self.action_values = None
        self.observation_space = gym.spaces.Box(low=0, high=float('Inf'), shape=(1,), dtype=np.float64)
        self.update_actions()
        self.n_actions = len(self.action_values)
        self.cvars = [[] for _ in range(self.n_actions)]
        self.cvar_cis = [[] for _ in range(self.n_actions)]
        self.means = [[] for _ in range(self.n_actions)]
        self.vars_ = [[] for _ in range(self.n_actions)]
        self.true_params = [[] for _ in range(self.n_actions)]
        self.soil_features = ['SLDR', 'SUM_WHC', 'SUM_SH2O', 'SUM_C']
        if not stateless:
            self.state_features = None
            self.get_state_features()
        self.y_kde = []
        self.y_kde_points = []
        self.set_seed()

    def check_for_dssat_auxiliary_files(self):
        """
        Checks that if action-making include cultivar choice that a valide cultivar list is available.
        Checks that if the random soil option is activated, that valide soil list is available.
        """
        if self.cultivar:
            assert 'cultivar_path' in self.dssat_param_dic
        if 'random_soil' in self.dssat_param_dic and self.dssat_param_dic['random_soil']:
            assert 'soil_path' in self.dssat_param_dic

    def set_seed(self, seed=None):
        """
        @param seed: seed to be set
        @type seed: integer
        """
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        random.seed(seed)
        self.dssat.set_seed(seed)

    def update_actions(self):
        """
        Generates the available actions in the environment
        """
        date_range = self.get_planting_range()
        if self.cultivar:
            action_list = [date_range, self.dssat.cultivars]
        elif self.sowing_date:
            action_list = [date_range, [self.dssat.ingeno]]
        else:
            action_list = [[self.dssat.pdate], [self.dssat.ingeno]]
        self.action_values = list(itertools.product(*action_list))
        self.action_space = gym.spaces.Discrete(len(self.action_values))

    def step(self, action_index, seed=None):
        """
        Make a one step ahead in the environment, and get the reward. Each time self.dssat.reset_random() is called,
        thus if random weather or soil are activated, it samples a new values of those parameters.
        @param action_index: the index of the action to be made
        @type action_index: integer in range(self.n_actions)
        @param seed: to be ignore
        @type seed: integer
        @return: state (DSSAT contextual information if stateless is False), reward (list of rewards), None, None
        @rtype: Tuple
        """
        self.dssat.reset_random()
        reward = self._get_reward(action_index)
        info = {'soil': self.dssat.id_soil, 'random_weather': self.dssat.random_weather,
                'random_soil': self.dssat.random_soil}
        episode_over = None
        state = None
        if not self.stateless:
            state = self.get_state()  # has to be retrieve after reward is computed
        return state, reward, episode_over, info

    def read_julian_date(self, julian_date):
        """
        Read a date in the julian format as a string and return a datetime object
        @param julian_date: a julian datetime in DSSAT format
        @type julian_date: string 'YYDDD'
        @return: the converted date
        @rtype: datetime object
        """
        date = datetime.datetime.strptime(julian_date, '%y%j').date()
        return date

    def get_first_day_of_the_week(self, date):
        """
        @param date: a date
        @type datetime object
        @return: the first day of the week for the considered date
        @rtype: datetime object
        """
        first_day = date - datetime.timedelta(days=date.isoweekday() % 7)
        return first_day

    def get_planting_range(self):
        """
        Get from an initial window planting date (self._planting_date) a list of (self.date_steps) successive planting
        dates spaced by the same number of days (self.date_steps). Insures that the dates are posterior to
        self.dssat.icdat that is to say the initial soil condition measure date.
        @return: an ensemble of planting dates in the julian format
        @rtype: list of strings in the format 'YYDDD'
        """
        icdat = self.read_julian_date(self.dssat.icdat)
        original_plating_date = self.read_julian_date(self.dssat._planting_date)
        date_steps = range(self.date_steps + 1)
        planting_date_range = [original_plating_date + datetime.timedelta(days=self.date_delta * date_step)
                               for date_step in date_steps]
        planting_date_range = np.array(planting_date_range)
        planting_date_range = planting_date_range[planting_date_range >= icdat]
        planting_date_range = [date.strftime('%y%j') for date in planting_date_range]
        return planting_date_range

    def get_state(self):
        """
        Get contextual information about DSSAT
        @return: state, an ensemble of I/O of DSSAT
        @rtype: a list of DSSAT parameters
        """
        state = []
        param_dics = self.dssat.get_context()
        if self.dssat.random_soil:
            state.extend(self._get_soil_state(param_dics['soil']))
        # if self.dssat.random_weather:
        # 	state.extend(self._get_historical_weather_state(param_dics['historical_weather']))
        if self.dssat.additional_data and param_dics['additional_data']:
            state.extend(self._get_additional_state(param_dics['additional_data']))
            state.extend(self.process_dic(param_dics['summary'], self.dssat.output_context_cols))
        return state

    def _get_soil_state(self, soil_param_dic):
        """
        Compute some statistics about soil
        @param soil_param_dic: soil parameters
        @type soil_param_dic: dictionary
        @return: soil statistics
        @rtype: list
        """
        SLDR = soil_param_dic['SLDR']
        SDUL = soil_param_dic['SDUL']
        SLLL = soil_param_dic['SLLL']
        SLB1 = soil_param_dic['SLB']
        SLB2 = SLB1.copy()  # bottom layers
        SLB2 = np.insert(SLB2, 0, 0)[:-1]  # top layers
        SLB_lower_than = SLB2 <= 50  # select all layer beginning before or equal to the threshold
        thicknesses = SLB1 - SLB2
        SBDM = soil_param_dic['SBDM']
        SLOC = soil_param_dic['SLOC']
        c_content = thicknesses * SBDM * SLOC * SLB_lower_than
        capacity = (SDUL - SLLL) * thicknesses  # cm3.cm-2
        initial_water_content = self.dssat.initial_H2O_factor * capacity  # + SLLL * thicknesses
        return [SLDR.sum(), capacity.sum(), initial_water_content.sum(), c_content.sum()]

    def _get_historical_weather_state(self, historical_weather_param_dic):
        """
        Get historical weather statistics
        @param historical_weather_param_dic: historical weather values
        @type historical_weather_param_dic: dictionary
        @return: historical weather statistics
        @rtype: list
        """
        monthly_averages_dic = historical_weather_param_dic['MONTHLY AVERAGES']
        keys = ['SAMN', 'XAMN', 'NAMN', 'RTOT', 'RNUM']
        values = self.process_dic(monthly_averages_dic, keys)
        return values

    def _get_additional_state(self, additional_data_dic):
        """
        Utility function to concatenate state dictionary values
        @param additional_data_dic: a dictionary to be concatenated
        @type additional_data_dic: dictionary
        @return: 1 D values
        @rtype: list
        """
        keys = self.dssat.additional_data_cols
        values = self.process_dic(additional_data_dic, keys)
        return values

    @staticmethod
    def get_chunk_limits(lenght, n_chunk):
        """
        Gives indices of chunks to split a list in n_chunk parts
        :param lenght: len of the list to split
        :type lenght: int
        :param n_chunk: the number of chunks to get
        :type n_chunk: int
        :return: split indices' limits
        :rtype: np.array
        """
        dividend = lenght // n_chunk
        limits = [[dividend * i, dividend * (1+i)] for i in range(n_chunk - 1)]
        limits.append([dividend * (n_chunk - 1), lenght])
        return np.array(limits)

    def make_rseed_lists(self, n_chunk):
        """
        Split the rseed list into n_cores chunks
        :return: nothing
        :rtype: None
        """
        limits = self.get_chunk_limits(lenght=self.dssat.rseed_upper_bound + 1 - self.dssat.rseed_lower_bound,
                                       n_chunk=n_chunk)
        all_rseeds = np.array(range(self.dssat.rseed_lower_bound, self.dssat.rseed_upper_bound + 1))
        np.random.shuffle(all_rseeds)
        rseed_lists = [all_rseeds[limit[0]:limit[1]] for limit in limits]
        return rseed_lists

    def process_dic(self, dic, keys):
        """
        Utility function to concatenate dictionary values
        @param dic: dictionary to concataned
        @type dic: dictionary
        @param keys: keys for values to be concatenated
        @type keys: list
        @return: concatenated values
        @rtype: list
        """
        values = [dic[key] for key in keys]
        values = np.concatenate(values, axis=None)
        return values

    def get_state_features(self):  # keys could be stocked in attribute
        """
        Fills the DSSAT state for contextual information
        """
        state_features = []
        if self.dssat.random_soil:
            state_features.extend(self.soil_features)
        # if self.dssat.random_weather:
        # 	for month in range(1,13):
        # 		state_features.extend([f'SAMN_{month}', f'XAMN_{month}', f'NAMN_{month}', f'RTOT_{month}',
        # 		                       f'RNUM_{month}'])
        if self.dssat.additional_data:
            state_features.extend(self.dssat.additional_data_cols)
        if self.dssat.output_context:
            state_features.extend(self.dssat.output_context_cols)
        self.state_features = state_features

    def _translate_action(self, action):
        """
        Turns an action index into a meaningful action value for DssatEnv
        @param action: action index in range(self.n_actions)
        @type action: int
        @return: action value
        @rtype: list of strings
        """
        return self.action_values[action]

    def _get_reward(self, action_index):
        """
        Get a yield observation from an action index to be made
        @param action_index: action index in range(self.n_actions)
        @type action_index: int
        @return: a (normalized by self.eta_max) yield observation
        @rtype: float
        """
        action = self._translate_action(action_index)
        self.dssat.pdate = action[0]
        self.dssat.ingeno = action[1]
        if self.cultivar:  # default cultivar not in dict, to avoid error
            self.dssat.cname = self.dssat.cultivar_descriptions[self.dssat.ingeno]
        reward = self.dssat.step()
        if self.normalize:
            reward = reward / self.eta_max
        return reward

    def reset(self, seed=None):
        """
        Reset randomness of DSSAT observations and enventually set a new seed
        @param seed: the seed to be enventually set
        @type seed: int
        """
        self.set_seed(seed)
        self.dssat.reset_random()

    def load_samples(self, loading_path=None):
        """
        Loads saved sampling.
        @param loading_path: path of to the pickle of samples to be loaded
        @type loading_path: string
        @return: nothing
        @rtype: None
        """
        if loading_path is None:
            loading_path = self.sample_loading_path
        with open(loading_path, 'rb') as f:
            saved_sampling = pickle.load(f)
        self.env_param_dic_loaded_samples = saved_sampling['env_param_dic']
        self.dssat_param_dic_loaded_samples = saved_sampling['dssat_param_dic']
        samples = saved_sampling['samples']
        if self.restriction_list is not None:
            samples = np.array(samples)[self.restriction_list]
        self.samples = samples

    def get_dist_params(self, n_samples=None, saving_path=None, loading_path=None, restriction_list=None, cvar_ci=False,
                        compute_kdes=False, worker_pool=None):
        """
        Sample arms if saving_path is not None. Loads samples if loading_path is not None. Estimate DssatEnv amrs'
        distributions (means, variances, cvars).
        @param n_samples: the number of samples to be collected for each arm
        @type n_samples: integer
        @param saving_path: path to save the whole sampling (None if pre-computed samples to be loaded)
        @type saving_path: string
        @param loading_path: path to load pre-computed samples (None if samples to be collected)
        @type loading_path: string
        @param restriction_list: an eventual subset of all sampled arms
        @type restriction_list: list of booleans of shape (self.n_actions)
        @param cvar_ci: if 1 - self.delta -CVaR confidence intervals to be computed
        @type cvar_ci: bool
        @param worker_pool: a pool of joblib's Parallel to reused across processes
        @type worker_pool: joblib's Parallel class
        @return: nothing
        @rtype: None
        """
        assert loading_path is not None or saving_path is not None
        n_cores = mp.cpu_count()

        if loading_path:
            print('\n~~~ sample loading ~~~')
            self.sample_loading_path = loading_path
            self.load_samples()
            print('~~~ \sample loading ~~~')
        if saving_path:
            if not n_samples:
                n_samples = self.n_samples
            samples_per_core = n_samples // n_cores
            self.sample_loading_path = saving_path
            print('\n~~~ sampling stage starts ~~~')
            seeds = np.random.choice(range(10000000), size=n_cores, replace=False)
            rseed_lists = cycle(self.make_rseed_lists(n_cores))
            seeds = cycle(seeds)
            all_samples = []
            for action_index, _ in tqdm(enumerate(self.action_values)):
                arguments = [(action_index, samples_per_core, next(seeds), next(rseed_lists)) for _ in range(n_cores)]
                if worker_pool is None:
                    # with Parallel(n_jobs=-1, max_nbytes=None, timeout=None, verbose=100) as :
                    try:
                        parallel = Parallel(n_jobs=-1, max_nbytes=None, timeout=None, verbose=100)
                        res = parallel(delayed(self.parallel_sampling_)(argument) for argument in arguments)
                    finally:
                        get_reusable_executor().shutdown(wait=False)
                else:
                    res = worker_pool(delayed(self.parallel_sampling_)(argument) for argument in arguments)
                all_samples.append(res)
            all_samples = np.array(all_samples)
            self.samples = all_samples.reshape(all_samples.shape[:-2] + (-1,))  # concatenates two last dimensions
            if self.normalize:
                self.samples *= self.eta_max
            if saving_path is not None:
                with open(saving_path, 'wb') as f:
                    to_save = {'env_param_dic': self.env_param_dic,
                               'dssat_param_dic': self.dssat_param_dic,
                               'samples': self.samples}
                    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('~~~ sampling stage completed ~~~')

        print('~~~ statistics computation starts ~~~')
        for action_index in range(self.n_actions):
            self.cvars[action_index] = moment_empirical_cvar(self.samples[action_index], self.alpha)
            if cvar_ci:
                self.cvar_cis[action_index] = get_cvar_cis_left(self.samples[action_index], self.alpha, self.delta,
                                                                self.supp)
        self.means = np.mean(self.samples, axis=1)
        self.vars_ = np.var(self.samples, axis=1)
        if self.is_cvar:
            self.true_params = self.cvars
        else:
            self.true_params = self.means
        if compute_kdes:
            self.compute_kdes()
        print('~~~ statistics computation completed ~~~\n')

        if restriction_list is not None:
            self.restriction_list = restriction_list
            self.restriction_on_samples()
        del self.samples  # free up memory
        self.samples = None

    def set_alpha(self, alpha, worker_pool=None):
        """
        Changes the CVaR alpha value and recompute the CVaRs
        @param alpha: CVaR alpha value
        @type alpha: float
        @param worker_pool: a pool of joblib's Parallel to reused across processes
        @type worker_pool: joblib's Parallel class
        @return: nothing
        @rtype: None
        """
        self.alpha = alpha
        self.get_dist_params(loading_path=self.sample_loading_path, worker_pool=worker_pool)

    def set_eta_max(self, eta_max):
        """
        Changes yield upper bound
        @param eta_max: yield upper bound, kg/ha
        @type eta_max: float
        @return: nothing
        @rtype: None
        """
        self.eta_max = eta_max
        self.supp[-1] = eta_max

    def restriction_on_samples(self, restriction_list=None):
        """
        Restricts available actions to the ones given in restriction_list and update relevant attributes
        @param restriction_list: an eventual subset of all sampled armss
        @type restriction_list: list of booleans of shape (self.n_actions)
        @return: nothing
        @rtype: None
        """
        if restriction_list is None:
            restriction_list = self.restriction_list
        assert len(restriction_list) == self.n_actions

        def update(x):
            if x is not None:
                return np.array(x)[restriction_list]

        self.n_actions = np.array(restriction_list).sum()
        self.action_values = update(self.action_values)
        self.actions_index = range(self.n_actions)
        self.samples = update(self.samples)
        self.means = update(self.means)
        self.vars_ = update(self.vars_)
        self.true_params = update(self.true_params)
        self.cvars = update(self.cvars)
        self.cvar_cis = update(self.cvar_cis)
        self.action_space = gym.spaces.Discrete(len(self.action_values))
        dict_to_print = {action_index: {'action_value': action_value,
                                        'mean': mean,
                                        'cvar': cvar}
                         for action_index, action_value, mean, cvar
                         in zip(self.actions_index, self.action_values, self.means, self.cvars)}
        print('\n~ action restriction ~\n')
        pprint.pprint(dict_to_print)
        print('\n~~~~~~~~~~~~~~~~~~~~~~\n')

    def parallel_sampling_(self, args):
        """
        Function to be parallelized for arm sampling
        @param args: iterable containing (action_index, n_samples, seed)
        @type args: tuple or list
        @return: yield samples from the given action index
        @rtype: list of shape (self.n_samples)
        """
        try:
            action_index, n_samples, seed, rseed_list = args
            self.dssat.make_tmp_folder()
            self.set_seed(seed)
            self.dssat.set_rseed_iterator(rseed_list=rseed_list)
            samples = []
            for _ in range(n_samples):
                _, reward, _, _ = self.step(action_index, seed=seed)
                samples.extend(reward)
            return samples
        except Exception as e:
            logging.exception(e)
        finally:
            self.dssat.close()

    def compute_kdes(self):
        if self.samples is None:
            print('\n~~~ sample loading for kdes ~~~')
            self.load_samples()
            print('~~~ \sampled loading for kdes ~~~\n')
        for action_index in range(self.n_actions):
            print(f'~~ computing kde for action #{action_index} ~~')
            sample = self.samples[action_index]
            x_kde = np.linspace(start=self.supp[0], stop=self.supp[1], num=10000)
            kde_f = gaussian_kde(sample)
            y_kde = kde_f(x_kde)
            x_kde_points_step = self.supp[1] // 10
            x_kde_points = range(x_kde_points_step, self.supp[1], x_kde_points_step)
            y_kde_points = kde_f(x_kde_points)
            self.y_kde_points.append(y_kde_points)
            self.y_kde.append(y_kde)
            print(f'~~ /computing kde for action #{action_index} ~~')
        if self.samples is not None:
            del self.samples
            self.samples = None

    def render_env(self, saving_path=None, kde=True, hist=False, cvar_ci=None, title=None, fout='pdf', cvar_bars=True):
        """
        Plots and saves DSSAT environment with respective CVaRs at level self.alpha
        @param saving_path: path to save the plot
        @type saving_path: string
        @param kde: if distributions' gaussian kernel density estimations are plotted
        @type kde: bool
        @param hist: if sample stepwise normalized histogram are plotted
        @type hist: bool
        @param cvar_ci: if distributions' self.alpha cvars self.delta confidence intervals are plotted
        @type cvar_ci: boot
        @param title: environment plot title
        @type title: string
        @param fout: format to save the regret figures, e.g. 'png', 'pdf'
        @type fout: string
        @return: nothing
        @rtype: None
        """
        if cvar_ci:
            for action_index in range(self.n_actions):
                assert self.cvar_cis[action_index]  # assert that CVaR confidence intervals have been computed

        print('\n#### DSSAT RENDERING STAGE ###')

        if saving_path is None:
            saving_path = 'dssat_render'
        palette = sns.color_palette('colorblind')
        markers = ['o', '^', 's', 'p', 'X', 'D', 'd']
        fig, ax = plt.subplots()
        legend_elements = []

        if kde and len(self.y_kde) < self.n_actions:
            self.compute_kdes()

        if hist:
            print('\n~~~ sample loading for rendering stage ~~~')
            self.load_samples()
            print('~~~ \sampled loading for rendering stage ~~~\n')

        for action_index in range(self.n_actions):

            cvar_emp = self.true_params[action_index]
            mean = self.means[action_index]
            label = f'day: {self.action_values[action_index][0][2:]}'
            marker = markers[action_index]
            dash = (1, 0, 0, 0)
            color = palette[action_index]
            if hist:
                label_hist = None
                if not kde:
                    label_hist = label
                sample = self.samples[action_index]
                hist_ = ax.hist(sample, alpha=0.7, bins='auto', histtype='step', density=True,
                                color=color,
                                label=label_hist,
                                zorder=0)
                if not kde:
                    legend_elements.extend(hist_[-1])
            if kde:
                x_kde = np.linspace(start=self.supp[0], stop=self.supp[1], num=10000)
                y_kde = self.y_kde[action_index]
                y_kde_points = self.y_kde_points[action_index]
                kde_p = ax.plot(x_kde, y_kde, alpha=1, color=color, zorder=1, dashes=dash, linewidth=3)
                x_kde_points_step = self.supp[1] // 10
                x_kde_points = range(x_kde_points_step, self.supp[1], x_kde_points_step)
                points_kde = ax.plot(x_kde_points, y_kde_points, color=color, marker=marker,
                                     zorder=3, markersize=8, linestyle='')
                legend_element = Line2D([None], [None], label=label, dashes=dash, marker=marker,
                                        color=color, markersize=8, linewidth=3)
                legend_elements.append(legend_element)
            if cvar_bars:
                ax.axvline(x=cvar_emp, color=palette[action_index], linestyle='--', linewidth=3, alpha=1)
            if cvar_ci and self.cvar_cis[0]:
                c_min, c_max = self.cvar_cis[action_index]
                ax.axvline(x=c_min, color=palette[action_index], linestyle='dashed', linewidth=2, alpha=0.3)
                ax.axvline(x=c_max, color=palette[action_index], linestyle='dashed', linewidth=2, alpha=0.3)
        upper_bound = ax.axvline(x=self.supp[1], color='b', linestyle='dashdot', linewidth=2, label='yield upper bound')
        metric_legend_label = None
        if self.is_cvar and cvar_bars:
            metric_legend_label = f'CVaR @ {100 * self.alpha:.0f}%'
        elif not self.is_cvar:
            metric_legend_label = 'empirical mean'
        if metric_legend_label is not None:
            metric_legend = Line2D([None], [None], color='black', lw=2, label=metric_legend_label, linestyle='--')
            legend_elements.append(metric_legend)
        legend_elements.append(upper_bound)
        ax.set_xlabel('dry grain yield (kg/ha)')
        ax.set_ylabel('density')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.set_aspect('auto')
        ax.set_xlim([self.supp[0], 1.05 * self.supp[1]])
        ax.set_xbound(lower=self.supp[0], upper=1.05 * self.supp[1])
        if title is None:
            title = f'Empirical distributions estimated after #{self.n_samples:.0e} samples'
        plt.title(title)
        ax.legend(handles=legend_elements, loc='best')
        plt.tight_layout()
        plt.savefig(f'{saving_path}.{fout}', format=fout)
        plt.close()

        if self.samples is not None:
            del self.samples
            self.samples = None

        print('\n#### \DSSAT RENDERING STAGE ###\n')

    def close(self):
        pass

    def del_samples(self):
        """
        deletes loaded samples
        :return: None
        :rtype: None
        """
        del self.samples
        self.samples = None

def get_cvar_cis_left(samples, alpha, delta, supp):
    """
    Confidence intervals for quantile 0 to quantile alpha CVaR of a bounded real valued random variable.
    Adapted from https://arxiv.org/pdf/1901.00997.pdf
    @param samples: samples from the bounded real valued random variable
    @type samples: list of shape (n_samples)
    @param alpha: the level of 0 to alpha CVaR
    @type alpha: 0 <= float <= 1
    @param delta: confidence level for 1 - delta confidence interval
    @type delta: 0 <= float <= 1
    @param supp: bounded real valued random variable support
    @type supp: list [lower_support_bound, upper_support_bound]
    @return: alpha level CVaR confidence interval with confidence 1 - delta
    @rtype: list [lower_bound, upper_bound]
    """
    n = len(samples)
    sorted_samples = sorted(samples)
    diffs = np.diff(sorted_samples, prepend=supp[0], append=supp[1])
    b = np.sqrt(np.log(2 / delta) / (2 * n))

    term1_c_max = np.minimum([i / n - b for i in range(1, n + 1)], alpha)
    term2_c_max = diffs[1:]
    term_c_max = term1_c_max * term2_c_max
    c_max = supp[1] - 1 / alpha * term_c_max.sum()

    term1_c_min = np.minimum(1, np.maximum(0, np.minimum([i / n + b for i in range(n)], alpha)))
    term2_c_min = diffs[:-1]
    term_c_min = term1_c_min * term2_c_min
    c_min = sorted_samples[-1] - 1 / alpha * term_c_min.sum()
    return c_min, c_max


if __name__ == '__main__':
    pass
