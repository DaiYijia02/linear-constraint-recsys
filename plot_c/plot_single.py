import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from collections import defaultdict
from matplotlib.patches import Patch
import sys
from scipy.interpolate import interp1d
import itertools
from matplotlib.ticker import ScalarFormatter
import argparse
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--base_save_dir', default=f'{os.path.abspath(os.path.join(os.getcwd()))}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--job', type=str, choices=['tune', 'test'])
parser.add_argument('--print-commands', action='store_true')
parser.add_argument('--device', type=int, help='GPU to use')
parser.add_argument('--init_weights', choices=['default', 'zero'], default='default')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--algorithm', type=str)
parser.add_argument('--task', type=str)
args = parser.parse_args()

task_to_cost = {
    "kuai": "1._1.",
    "tv_audience": "0.01_0.01",
    "lastfm": "1._1.",
}

def main() -> int:
    fig, axes= plt.subplots(nrows = 1, ncols = 5, figsize=(10,2))     
    markevery_dict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1,8:1,9:1,10:1}
    marker_dict={0:'D', 1:'P', 2:'X', 3:'o', 4:'P', 5:'h', 6:6, 7:11,8:10, 9:9,10:7}
    # styles={1:'-', 7:'--', 2:':'}
    styles={0:'-', 1:'--', 2:':'}

    # label={0:'asgd_onlinebpcnoerror',\
    #        1:'lpnc',\
    #        2:'smpct_is',\
    #        3:'sgd_onlinebpc',\
    #        4:'rms_onlinebpc',
    #        5:'adagrad_onlinebpc',
    #        6:'adam_onlinebpc',
    #        7:'onlinebpcnoerror',\
    #        8:'rms_onlinebpcnoerror', # smpct_avg
    #        9:'adagrad_onlinebpcnoerror',
    #        10:'adam_onlinebpcnoerror'
    # } 

    # plot_label={0:'CA_NoError_ASGD',\
    #        1:'CO',\
    #        2:'CA',\
    #        3:'CA_SGD',\
    #        4:'CA_RMS',\
    #        5:'CA_ADAGRAD',
    #        6:'CA_ADAM',
    #        7:'CA',
    #        8:'CA_NoError_RMS',
    #        9:'CA_NoError_ADAGRAD',
    #        10:'CA_NoError_ADAM'
           
    # }
    label={0:'oracle',\
           1:'bpc',\
           2:'onlinebpcnoerror',\
    } 

    plot_label={0:'oracle',\
           1:'BPC',\
           2:'CA',\
    }
    y_max_util = []
    #for col in [4,2,5,10,11]:#range(4):
    # for col in [1, 7, 2]:
    for col in [0, 1, 2]:

        #x = [0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100]
        #x = [1..]
        #for gamma in [.1]:
        #for gamma in ['None']:

        gamma = .1
        #for lr in [0.001, 0.01, 0.1]:
        hyparams = [(lr, gamma)
            for lr in [0.001, 0.01, 0.1, 1., 10, 100]
            for gamma in [10, 1, 0.1, .01]]
        hyparams = [(lr, gamma)
            for lr in [ 0.1]
            for gamma in [1]]
        for (lr, gamma) in hyparams:
            y_dcg = []
            y_weighted_objective = []
            y_unsat_mean = []
            y_unsat_sum = []
            y_time = []
            y_delta = []
            y_eposure = []
            y_state = []
            y_delta = []
            y_per_step_exposure = []

            # if (col in [7] and gamma != 10):
            #     continue 
            # if (col in [1] and lr != 1):
            #     continue 
            # if (col in [2] and gamma !=10):
            #     continue

            #for c in [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]:
            #for c in ["0.0001_0.001", "0.001_0.001", "0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]:
            # for c in ["10._10."]:
            # for c in ["1._1."]:
            for c in [task_to_cost[args.task]]:

                if label[col] == 'bpc':
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [('None', c, init_, b, bo, gamma, 'online_relevance') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] == 'oracle':
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [('None', c, init_, b, bo, gamma, 'online_relevance') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] in ['smpct_is', 'smpct_is_error', 'smpct_is_approx_hinge']:
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [(lr, c, init_, b, bo, gamma, 'offline_relevance')\
                        for b in [1]
                        for bo in [10]
                        for init_ in ['zero']]
                        #for lr in ['None']]
                elif label[col] in ['smpct_is_exact_hinge', 'smpct_exact_c']:
                    args.algorithm = label[col] 
                    #gamma = 1
                    params = [(lr, c, init_, b, bo, gamma, 'online_relevance')\
                        for b in [1]
                        for bo in [1]
                        for init_ in ['zero']]
                        #for lr in ['1.0']]
                elif label[col] in ['lpc', 'lpnc']:
                    args.algorithm = label[col] 
                    #gamma =1 
                    params = [('None', c, init_, b, bo, gamma, 'online_relevance')\
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] in ['onlinebpcnoerror']:
                    args.algorithm = label[col] 
                    gamma = 1
#                    if label[col] in [ 'asgd_onlinebpc', 'sgd_onlinebpc', 'adagrad_onlinebpc', 'adam_onlinebpc']:
#                        lr = 0.1
#                    if label[col] in ['rms_onlinebpc']:
#                        lr = 0.01
#                    if label[col] == 'onlinebpc':
#                        lr =1. 
#                    elif label[col] == 'onlinebpcnoerror':
#                        lr =0.1
                    params = [(lr, c, init_, b, bo, gamma, 'online_relevance') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['zero']
                        #for init_ in ['one']]
                        for lr in [0.0001, 0.001, 0.01, 0.1, 1., 10, 100, 1000]]
                else:
                    raise Exception(f'Unknow alg: {label[col]}') 
        
                # unsatisfaction_mean = -np.inf
                # unsatisfaction_std = np.inf
                # dcg = -np.inf
                weighted_objective = -np.inf
                file = None
        
                for param in params:
                    (lr, c, init_, b, bo, gamma, r_type) = param
                    name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                    name = name.replace("/","_")
        
                    try:
                        df = pd.read_pickle(f'results/{args.task}/{name}_dev.pkl')
                    except:
                        print(f'results/{args.task}/{name}_dev.pkl')
       
                    # if math.isclose(df.iloc[-1]['Unsatisfaction Mean'] , 0) and\
                    #    math.isclose(df.iloc[-1]['Unsatisfaction Std'] , 0) and \
                    #    df.iloc[-1]['Sum DCG'] > dcg:
                    if df.iloc[-1]['Weighted Objective'] > weighted_objective:
                        #    dcg =  df.iloc[-1]['Sum DCG']
                           weighted_objective = df.iloc[-1]['Weighted Objective']
                           file = f'results/{args.task}/{name}.pkl'
        
                if file == None:
                    print(f'{args.algorithm} -- unsatisfied')
                    for param in params:
                        (lr, c, init_, b, bo, gamma, r_type) = param
                        name = f"{args.algorithm}_" + "_".join([str(x) for x in param ])
                        name = name.replace("/","_")
            
                        df = pd.read_pickle(f'results/{args.task}/{name}.pkl')
                        if df.iloc[-1]['Unsatisfaction Mean'] > unsatisfaction_mean and\
                           df.iloc[-1]['Unsatisfaction Std'] < unsatisfaction_std: 
                               unsatisfaction_mean = df.iloc[-1]['Unsatisfaction Mean']
                               unsatisfaction_std = df.iloc[-1]['Unsatisfaction Std']
                               file = f'results/{args.task}/{name}.pkl'
                else:
                    print(f'Best {args.algorithm} file: {file} {weighted_objective}')
                    # print(f'Best {args.algorithm} file: {file}i {dcg}')
        
                df = pd.read_pickle(file)
     
                #y_dcg.append(df['Sum DCG'])
                y_dcg = df['Sum DCG']
                #y_weighted_objective.append(df['Weighted Objective'])
                y_weighted_objective = df['Weighted Objective']
                #y_unsat_mean.append(df['Unsatisfaction Mean'])
                y_unsat_mean = df['Unsatisfaction Mean']
                #y_unsat_sum.append(df['Unsatisfaction Sum'])
                y_unsat_sum = df['Unsatisfaction Sum']
                # ...
                y_state = np.stack(np.array(df['per_step_exposure']))
                # ...
                y_tau = np.stack(np.array(df['tau']))
                # ...
                y_per_step_exposure = np.stack(np.array(df['per_step_exposure']))
                #y_delta.append(df['
                try:
                    y_time.append(df['elapsed_time'].iloc[-1]/60.)
                except:
                    import pdb; pdb.set_trace()

            #axes[0].set_title(f'{args.algorithm}', fontsize='small')

            # Plot DCG 
            x = range(len(y_dcg))
            # axes[0].plot(x, y_dcg, zorder=10, linestyle=styles[col], label=f'{plot_label[col]}_{lr}_{gamma}' if col >= 0 else f'{plot_label[col]}')
            axes[0].plot(x, y_dcg, zorder=10, linestyle=styles[col], label=f'{plot_label[col]}' if col >= 0 else f'{plot_label[col]}')
            axes[0].set_title('DCG', fontsize='small')
            axes[0].set_xlabel(r'steps', fontsize='small')
            #axes[0].set_xscale('log')
            #axes[0].set_xlim(0, 100)
            #axes[0].set_xticks(np.arange(.001, 100, step=10))
    
#            # Plot Unsatisfaction Mean 
#            axes[1].plot(x, y_unsat_mean, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
#            axes[1].set_title('Unsat. Mean', fontsize='small')
#            axes[1].set_xlabel('c penality', fontsize='small')
#            axes[1].set_xscale('log')
    
            # Plot Unsatisfaction Sum 
            axes[1].plot(x, y_unsat_sum, linestyle=styles[col])
            axes[1].set_title(r'Violation Cost', fontsize='small')
            axes[1].set_xlabel(r'steps', fontsize='small')
            #axes[1].set_xscale('log')
            #axes[1].set_ylim(-0.5, 0.5)
    
            # Plot Weighted Objective
            axes[2].plot(x, y_weighted_objective, linestyle=styles[col])
            axes[2].set_title(r'Weighted Objective', fontsize='small')
            axes[2].set_xlabel(r'steps', fontsize='small')
            #axes[2].set_xscale('log')

            # Exposure
            axes[3].plot(x, y_tau[:,0], color='black')
            axes[3].plot(x, np.cumsum(y_state[:,0]), linestyle=styles[col])
            axes[3].set_title(r'C1: (1+tau)/T vs. Exposure', fontsize='small')
            axes[3].set_xlabel(r'steps', fontsize='small')
            #axes[3].set_xscale('log')


            # Exposure
            axes[4].plot(x, y_tau[:,1], color='black')
            axes[4].plot(x, np.cumsum(y_state[:,1]), linestyle=styles[col])
            axes[4].set_title(r'C2: (1+tau)/T vs. Exposure', fontsize='small')
            axes[4].set_xlabel(r'steps', fontsize='small')
            #axes[4].set_xscale('log')

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, ncol=8, frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(f'single_{args.task}.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
