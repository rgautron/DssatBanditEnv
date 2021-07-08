"""
This file contains functions to run replicated experiments of CVaR bandits under DSSAT.
It allows to save regret values, plot the environment, and plot regret figures.
"""
import faulthandler

faulthandler.enable()

import matplotlib
import logging

matplotlib.use('Agg')
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import seaborn as sns

sns.set_context("paper")

from cvarBandits.bandits import bandit_cvar
from gym_dssat.envs.dssat_env import DssatEnv
from tqdm import tqdm
import multiprocessing as mp
import joblib
import contextlib
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

import random
import numpy as np
from itertools import cycle

np.seterr(divide='raise')
import os, sys
import shutil
import glob
import itertools
import pathlib
import copy
import pdb
import time
import pickle
import pprint
import tempfile
import json

global_seed = 123

np.random.seed(global_seed)
random.seed(global_seed)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def average_score_problems(env, alphas, bandit_dicts, bandit_names, replications, horizon, init_pulls,
                           render_env=False, render_saving_path_prefix=None, y_label_regret=None, q_high=.95, q_low=.05,
                           global_seed=1234, saving_id=None, regret_plot_saving_prefix=None,
                           regret_values_saving_prefix=None, render_regrets=True, worker_pool=None, exp_nb=None):
    """
    Computes bandits' regret for multiples values of CVaR's alpha parameter in DSSAT environment.
    Save regret values, enventually environment and regret plots.
    @param env: the bandit environment
    @type env: DSSAT class
    @param alphas: alpha CVaR values for the environment
    @type alphas: list of floats
    @param bandit_dicts: bandit instances and parameters
    @type bandit_dicts: nested dictionaries
    @param bandit_names: bandit names for saving
    @type bandit_names: list of strings
    @param replications: number of replications to be performed
    @type replications: integer
    @param horizon: experiment horizon
    @type horizon: integer
    @param init_pulls: bandit initial pull number
    @type init_pulls: integer
    @param render_env: if the environment figures are plotted
    @type render_env: boolean
    @param render_saving_path_prefix: the folder to save environment plots
    @type render_saving_path_prefix: string
    @param y_label_regret: y axis label of regret plots
    @type y_label_regret: string
    @param q_high: upper quantile for confidence envelops in regret plots
    @type q_high: q_low <= float <= 1
    @param q_low: lower quantile for confidence envelops in regret plots
    @type q_low: 0 <= float <= q_high
    @param global_seed: the seed of the experiment
    @type global_seed: integer
    @param saving_id: a unique id for saving objects
    @type saving_id: string
    @param regret_plot_saving_prefix: the folder to save regret plots
    @type regret_plot_saving_prefix: string
    @param regret_values_saving_prefix:  the folder to save regret values
    @type regret_values_saving_prefix: string
    @param render_regrets: if regret figures are plotted
    @type render_regrets: bool
    @param worker_pool: a pool of joblib's Parallel to reused across processes
    @type worker_pool: joblib's Parallel class
    @param exp_nb: the number of experiment to be performed for saving identifier only
    @type exp_nb: int
    @return: nothing
    @rtype: None
    """
    for index, alpha in enumerate(alphas):
        print(f'\n\n ~~~ STARTING FOR ALPHA {alpha:.2f} HORIZON {horizon:.0e} ~~~ \n')
        print(f'alpha = {alpha}: {index + 1} out of {len(alphas)}')
        try:
            env.set_alpha(alpha=alpha, worker_pool=worker_pool)
        except Exception as e:
            logging.exception(e)
        render_saving_path = f'{render_saving_path_prefix}{saving_id}_dssat_render_N_{env.n_samples}_' \
                             f'A_{100 * alpha:.0f}_100_E_{exp_nb}'
        if render_env:
            env.render_env(saving_path=render_saving_path)
        mean_cum_regrets, quantiles_cum_regrets = compute_multiple_regrets(env=env,
                                                                           bandit_dicts=bandit_dicts,
                                                                           replications=replications,
                                                                           horizon=horizon,
                                                                           bandit_names=bandit_names,
                                                                           q_high=q_high,
                                                                           q_low=q_low,
                                                                           init_pulls=init_pulls,
                                                                           alpha=alpha,
                                                                           global_seed=global_seed,
                                                                           saving_id=saving_id,
                                                                           worker_pool=worker_pool,
                                                                           exp_nb=exp_nb,
                                                                           regret_values_saving_prefix=regret_values_saving_prefix)
        if render_regrets:
            plot_multiple_regrets(mean_cum_regrets=mean_cum_regrets,
                                  quantiles_cum_regrets=quantiles_cum_regrets,
                                  bandit_names=bandit_names,
                                  horizon=horizon,
                                  replications=replications,
                                  init_pulls=init_pulls,
                                  alpha=alpha,
                                  y_label=y_label_regret,
                                  q_high=q_high,
                                  q_low=q_low,
                                  saving_id=saving_id,
                                  EXP_NB=exp_nb,
                                  regret_plot_saving_prefix=regret_plot_saving_prefix)
        print(f'\n\n ~~~ DONE FOR ALPHA={alpha} HORIZON={horizon:.0e} ~~~ \n')


def compute_multiple_regrets(env, bandit_dicts, replications, horizon, bandit_names, init_pulls, alpha, q_low=.05,
                             q_high=.95,
                             global_seed=1234, saving_id=None, regret_values_saving_prefix=None, worker_pool=None,
                             exp_nb=None):
    """
    Computes bandits' regret for a given CVaR alpha value in DSSAT environment. enventually saves regret figures.
    @param env: the bandit environment
    @type env: DssatEnv class
    @param bandit_dicts: bandit instances and parameters
    @type bandit_dicts: nested dictionaries
    @param replications: number of replications to be performed
    @type replications: integer
    @param horizon: experiment horizon
    @type horizon: integer
    @param bandit_names: bandit names for saving
    @type bandit_names: list of strings
    @param init_pulls: bandit initial pull number
    @type init_pulls: integer
    @param alpha: alpha CVaR value for the environment
    @type alpha: float
    @param q_low: lower quantile for confidence envelops in regret plots
    @type q_low: 0 <= float <= q_high
    @param q_high: upper quantile for confidence envelops in regret plots
    @type q_high: q_low <= float <= 1
    @param global_seed: the seed of the experiment
    @type global_seed: integer
    @param saving_id: a unique id for saving objects
    @type saving_id: string
    @param regret_values_saving_prefix:
    @type regret_values_saving_prefix:
    @param worker_pool: a pool of joblib's Parallel to reused across processes
    @type worker_pool: joblib's Parallel class
    @param exp_nb: the number of experiment to be performed for saving identifier only
    @type exp_nb: int
    @return: mean_cum_regrets (list of averaged cumulative bandit regrets),  quantiles_cum_regrets
    (list of quantile values of cumulative bandit regrets)
    @rtype:  list of shape (n_bandits, replications, horizon), list of [quantiles_low, quantile_high] each one being of shape
    (n_bandits, replications, horizon)
    """
    try:
        bandit_list = [bandit_dic['instance'](env, **bandit_dic['params']) for bandit_dic in bandit_dicts]
        mean_cum_regrets = []
        quantiles_cum_regrets = []
        raw_results = {}
        seeds = np.random.choice(range(10000000), size=replications, replace=False)
        seeds = cycle(seeds)
        rseed_lists = cycle(env.make_rseed_lists(replications))
        for bandit, bandit_name in zip(bandit_list, bandit_names):
            arguments = [(bandit, horizon, next(seeds), env, bandit_dicts, next(rseed_lists))
                         for _ in range(replications)]
            with tqdm_joblib(tqdm(desc='Experiment replications', total=replications)) as progress_bar:
                if worker_pool is None:
                    # with Parallel(n_jobs=-1, max_nbytes=None, timeout=None, verbose=100) as parallel:
                    try:
                        parallel = Parallel(n_jobs=-1, max_nbytes=None, timeout=None, verbose=100)
                        raw_result = parallel(delayed(parallel_regret)(argument) for argument in arguments)
                    finally:
                        get_reusable_executor().shutdown(wait=False)
                else:
                    raw_result = worker_pool(delayed(parallel_regret)(argument) for argument in arguments)
            raw_results[bandit_name] = raw_result
            mean_cum_regrets.append(np.mean(raw_result, axis=0))
            quantiles_low = np.quantile(raw_result, q_low, axis=0, )
            quantiles_high = np.quantile(raw_result, q_high, axis=0)
            quantiles_cum_regrets.append([quantiles_low, quantiles_high])
        # print(f'means: {np.array(mean_cum_regrets)[:, -10:]}')
        # print(f'raw results :{[np.array(raw_results[name])[-1][-10:] for name in bandit_names]}')
        save_name = f'{regret_values_saving_prefix}{saving_id}_regret_values_R_{replications}_H_{horizon}_A_' \
                    f'{100 * env.alpha:.0f}_100_E_{exp_nb}.pkl'
        with open(save_name, 'wb') as f_:
            dic_to_save = {'mean_cum_regrets': mean_cum_regrets,
                           'quantiles_cum_regrets': quantiles_cum_regrets,
                           'bandit_names': bandit_names,
                           'horizon': horizon,
                           'q_low': q_low,
                           'q_high': q_high,
                           'replications': replications,
                           'raw_results': raw_results,
                           'init_pulls': init_pulls,
                           'alpha': alpha
                           }
            pickle.dump(dic_to_save, f_, protocol=pickle.HIGHEST_PROTOCOL)
        return mean_cum_regrets, quantiles_cum_regrets
    except Exception as e:
        logging.exception(e)


def parallel_regret(args):
    """
    Function to be parallelized for bandit regret computation
    @param args: iterable containing (bandit, horizon, seed, env, bandit_dicts)
    @type args: tuple or list
    @return: cumulative bandit regret
    @rtype: list of shape (horizon)
    """
    bandit, horizon, seed, env, bandit_dicts, rseed_list = args
    try:
        bandit.env.dssat.make_tmp_folder()  # each copy of the original instance to have its own temporary folder
        # in each core
        bandit.set_seed(seed)
        bandit.env.set_seed(seed)
        bandit.reset()
        env.dssat.set_rseed_iterator(rseed_list=rseed_list)
        for step in range(horizon - bandit.init_pulls * bandit.n_actions):
            bandit.step()
        regret = bandit.get_regret()
        return regret
    except Exception as e:
        logging.exception(e)
    finally:
        bandit.env.dssat.close()  # delete the temporary folder created before


def plot_multiple_regrets(mean_cum_regrets, quantiles_cum_regrets, bandit_names, horizon, replications, init_pulls,
                          alpha,
                          y_logscale=False, y_label=None, x_logscale=False, x_label=None, q_high=.95, q_low=.05,
                          saving_path=None, saving_id=None, regret_plot_saving_prefix=None, title=None, fout='pdf',
                          EXP_NB=None, *args, **kwargs):
    """
    Plots and saves bandits' cumulative regrets from compute_multiple_regrets function.
    @param mean_cum_regrets: list of averaged cumulative bandit regrets
    @type mean_cum_regrets: list of shape (n_bandits, replications, horizon)
    @param quantiles_cum_regrets: list of quantile values of cumulative bandit regrets
    @type quantiles_cum_regrets:  list of [quantiles_low, quantile_high] each one being of shape
    (n_bandits, replications, horizon)
    @param bandit_names: bandit names for saving
    @type bandit_names: list of strings
    @param horizon: experiment horizon
    @type horizon: integer
    @param replications: number of replications to be performed
    @type replications: integer
    @param init_pulls: bandit initial pull number
    @type init_pulls: integer
    @param alpha: alpha CVaR value for the environment
    @type alpha: float
    @param y_logscale: if y axis is scaled in log
    @type y_logscale: bool
    @param y_label: y label for the regret plot
    @type y_label: string
    @param x_logscale: if x axis is scaled in log
    @type x_logscale: bool
    @param x_label: x label for the regret plot
    @type x_label: string
    @param q_high: upper quantile for confidence envelops in regret plots
    @type q_high: q_low <= float <= 1
    @param q_low: lower quantile for confidence envelops in regret plots
    @type q_low: 0 <= float <= q_high
    @param saving_path: where the regret figures are saved
    @type saving_path: string
    @param saving_id: a unique id for saving objects
    @type saving_id: string
    @param regret_plot_saving_prefix: the folder to save regret plots
    @type regret_plot_saving_prefix: string
    @param title: title of the bandits' regret figure
    @type title: string
    @param fout: format to save the regret figures, e.g. 'png', 'pdf'
    @type fout: string
    @param EXP_NB: the number of experiment to be performed for saving identifier only
    @type EXP_NB: int
    @return: nothing
    @rtype: None
    """
    try:
        print('\n#### REGRET RENDERING STAGE ###')
        if saving_path is None:
            saving_path = f'{regret_plot_saving_prefix}{saving_id}_regret_plot_R_{replications}_H_{horizon}_A_' \
                          f'{100 * env.alpha:.0f}_100_E_{EXP_NB}'
        # Dorian's setting
        all_colors = np.array([('navy', 'lightblue', 'steelblue'), ('darkorange', 'navajowhite', 'orange'),
                               ('darkgreen', 'palegreen', 'forestgreen')])
        dashes = ['dashed', 'solid', 'dotted']
        markers = ['*', 'X', '^']
        # custom_colors = ['#014940', '#8d5524', '#370031', '#6846fc', '#83757e', '#909ffc']
        # palette_lines = sns.color_palette(n_colors=len(bandit_names))
        palette_lines = all_colors[:, 0]
        palette_fill = all_colors[:, 1]
        fig, ax = plt.subplots()
        legend_elements = []
        # dashes = get_custom_dashes(len(bandit_names))
        for index, (bandit_name, mean_regret, quantiles_cum_regret, dash) in enumerate(
                zip(bandit_names, mean_cum_regrets,
                    quantiles_cum_regrets, dashes)):
            color_fill = palette_fill[index]
            marker = markers[index]
            alpha_q = 1
            linewidth = 2
            if index == 0:
                zorder = 2
            else:
                zorder = 1
            quantiles_cum_regret_low, quantiles_cum_regret_high = quantiles_cum_regret
            x = np.array(range(1, len(mean_regret) + 1))
            line = ax.plot(x, mean_regret, color=palette_lines[index], linewidth=linewidth, linestyle=dash,
                           zorder=zorder + 2)
            x_points_step = horizon // 10
            x_points = range(x_points_step, horizon, x_points_step)
            common_x_indexes = [i for i, x in enumerate(x) if x in x_points]
            y_points = mean_regret[common_x_indexes]
            points = ax.plot(x_points, y_points, color=palette_lines[index], marker=marker, zorder=zorder + 2,
                             linestyle='', markersize=10, lw=3)
            legend_element = Line2D([None], [None], lw=2, label=bandit_name, linestyle=dash, marker=marker,
                                    linewidth=linewidth, color=palette_lines[index], markersize=10)
            legend_elements.append(legend_element)
            ax.fill_between(x=x, y1=quantiles_cum_regret_low, y2=quantiles_cum_regret_high, color=color_fill,
                            zorder=zorder - 1, alpha=alpha_q)
            ax.plot(x, quantiles_cum_regret_low, color=color_fill, linewidth=1, linestyle=dash, zorder=zorder)
            ax.plot(x, quantiles_cum_regret_high, color=color_fill, linewidth=1, linestyle=dash, zorder=zorder)
        if x_logscale:
            ax.set_xscale('log')
        if y_logscale:
            ax.set_yscale('log')
        if x_label is None:
            x_label = 'time step t'
            if x_logscale:
                x_label = f'{x_label[:-1]} log(t)'
        ax.set_xlabel(x_label)
        if y_label is not None:
            if y_logscale:
                y_label = f'log {y_label}'
            ax.set_ylabel(y_label)
        patch = Patch(facecolor='black', edgecolor=None, label=f'{q_low:.02f} to {q_high:.02f} quantile range',
                      alpha=.2)
        legend_elements.append(patch)
        # ax.grid(axis='both')
        ax.legend(handles=legend_elements, loc='best')
        if title is None:
            title = r'Averaged over #{replications} replications for $\alpha={alpha:.0f}$%' \
                .format(replications=replications, alpha=100 * alpha)
            plt.title(title)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{saving_path}.{fout}', format=fout)
        plt.close(fig)
        print('#### \REGRET RENDERING STAGE ###\n')
    except Exception as e:
        logging.exception(e)


def get_custom_dashes(n):
    """
    Generates n custom dashes tuples for matlplotlib, first one is plain
    @param n: the number of custom dashed to be generated
    @type n: integer
    @return: ensemble of n custom dashed
    @rtype: list of shape (n,4)
    """
    dashes = [[1, 0, 0, 0]]
    for index in range(n - 1):
        dashes.append([1 + index, 1 + index, 3 + index, 1 + index])
    return dashes


def save_pprint_dicts(dict, saving_path='pprint_dicts_save.txt'):
    """
    Utility to save a configuration dictionary in pprint fashion
    @param dict: configuration dictionary to be saved
    @type dict: dictionary
    @param saving_path: path for the configuration to be saved
    @type saving_path: string
    @return: nothing
    @rtype: None
    """
    with open(saving_path, 'wt') as out:
        pprint.pprint(dict, stream=out)


################## MAIN ##################


if __name__ == '__main__':
    ######### EXP CONFIG #########
    args_from_shell = False

    verbose = False

    print(f'\n\n !!! ARGS FROM SHELL: {args_from_shell} !!! \n\n')

    if args_from_shell:
        exp_nb = int(sys.argv[1])  # the number of the experiment to be run, defined by exp_dicts
        horizon = int(sys.argv[2])  # the bandit horizon of the experiment to be run
        replications = int(sys.argv[3])  # the number of replications of the experiment to be run
        sampling = eval(sys.argv[4])  # if DSSAT arm sampling has to be performed
        n_samples = int(sys.argv[5])  # the number of samples by arm for DSSAT arm sampling

        arg_from_shell_dict = {
            'exp_nb': exp_nb,
            'horizon': horizon,
            'replications': replications,
            'sampling': sampling,
            'n_samples': n_samples,
        }
        pprint.pprint(arg_from_shell_dict)
        time.sleep(10)
    else:
        horizon = int(1e4)
        replications = 1040
        exp_nb = 1
        sampling = True
        n_samples = 1 * int(1e6)

    init_pull = 1

    alphas = [.05, .2, .8]

    render_env = True
    render_regrets = True

    """
    restriction_list: subset of actions to consider
    eta_max: yield upper bound
    """

    exp_dicts = {
        1: {'restriction_list': [True, True, True, True, False, False, False],
            'eta_max': 10000
            },
        2: {'restriction_list': [True, True, True, True, False, False, False],
            'eta_max': 30000
            },
        3: {'restriction_list': [True, True, True, True, True, True, True],
            'eta_max': 10000
            }
    }

    ######### EXP UTILS #########
    if verbose:
        mpl = mp.log_to_stderr()
        mpl.setLevel(logging.INFO)

    restriction_list = exp_dicts[exp_nb]['restriction_list']
    eta_max = exp_dicts[exp_nb]['eta_max']

    exp_configs_path = './exp_configs/'
    render_saving_path_prefix = './env_figs/'
    regret_plot_saving_prefix = './regret_figs/'
    regret_values_saving_prefix = './regret_values/'

    for folder_name in [render_saving_path_prefix, exp_configs_path, regret_plot_saving_prefix,
                        regret_values_saving_prefix]:
        path = pathlib.Path(folder_name)
        path.mkdir(exist_ok=True)

    ######### ENV CONFIG #########
    date_steps = 6
    date_delta = 15
    id_soil = 'HC_GEN0027'

    dssat_param_dic = {
        'fileX_prefix': 'UFGA8201',
        'output': 'HWAM',
        'id_soil': id_soil,
        'sdate': '82056',
        'icdat': '82056',
        'planting_date': '82057',
        'random_weather': True,
        'ingeno': 'PC0005',
        'files_prefix': './dssat_files/',
        'experiment_number': exp_nb,
        'seed': global_seed,
    }

    env_param_dic = {
        'is_cvar': True,
        'stateless': True,
        'date_steps': date_steps,
        'date_delta': date_delta,
        'n_samples': n_samples,
        'eta_max': eta_max,
        'seed': global_seed
    }

    env_instance = DssatEnv

    ######### BANDIT CONFIG #########
    bandit_dics = [
        {'instance': bandit_cvar.B_CVTS},
        {'instance': bandit_cvar.BROWN_UCB},
        {'instance': bandit_cvar.CVAR_UCB},
    ]

    bandit_names = [
        "B-CVTS",
        "U-UCB",
        "CVaR-UCB",
    ]

    for param_dic in bandit_dics:
        if 'params' not in param_dic:
            param_dic['params'] = {}
        param_dic['params']['init_pulls'] = init_pull
        param_dic['params']['horizon'] = horizon

    ######### SAMPLE PRECOMPUTING #########
    saving_id = time.strftime("%Y%m%d_%H%M%S")

    sample_saving_path = f'{saving_id}_dssat_samples_N_{n_samples}_ST_{date_steps}_D_{date_delta}_SL_{id_soil}_' \
                         f'E_{exp_nb}.pkl'
    sample_loading_path = '20210211_205447_dssat_samples_N_1000000_ST_6_D_15_SL_HC_GEN0027_E_1.pkl'
    # sample_loading_path = None

    print(f'\n\n############ CONFIG ############\n\n')
    config_dic_to_print = {
        'sampling': sampling,
        'render_env': render_env,
        'render_regrets': render_regrets,
        'n_samples': n_samples,
        'horizon': horizon,
        'replications': replications,
        'id_soil': id_soil,
        'experiment': exp_nb,
        'restriction_list': restriction_list,
        'eta_max': env_param_dic['eta_max']
    }
    pprint.pprint(config_dic_to_print)
    time.sleep(10)

    exp_configs_saving_path = f'{exp_configs_path}{saving_id}_E_{exp_nb}.config'
    save_pprint_dicts(dict={'exp_config': config_dic_to_print,
                            'env_param_dic': env_param_dic,
                            'dssat_param_dic': dssat_param_dic,
                            },
                      saving_path=exp_configs_saving_path)
    print(f'\n\n############ \CONFIG ############\n\n')

    try:
        env = env_instance()
        env._init_(dssat_param_dic=dssat_param_dic, env_param_dic=env_param_dic, **env_param_dic)
        if sampling:
            env.get_dist_params(saving_path=f'{env.dssat.files_prefix}{sample_saving_path}',
                                restriction_list=restriction_list,
                                compute_kdes=render_env)
        else:
            env.get_dist_params(loading_path=f'{env.dssat.files_prefix}{sample_loading_path}',
                                restriction_list=restriction_list,
                                compute_kdes=render_env)

        ######### EXP LAUCHN #########
        y_label_regret = 'empirical yield regret (kg/ha)'

        average_score_problems(env=env,
                               alphas=alphas,
                               bandit_dicts=bandit_dics,
                               bandit_names=bandit_names,
                               replications=replications,
                               horizon=horizon,
                               init_pulls=init_pull,
                               render_env=render_env,
                               render_saving_path_prefix=render_saving_path_prefix,
                               y_label_regret=y_label_regret,
                               global_seed=global_seed,
                               saving_id=saving_id,
                               regret_plot_saving_prefix=regret_plot_saving_prefix,
                               regret_values_saving_prefix=regret_values_saving_prefix,
                               render_regrets=render_regrets,
                               exp_nb=exp_nb)
    except Exception as e:
        logging.exception(e)

    finally:
        env.dssat.close()  # removing temporary folder
