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
args = parser.parse_args()

def main() -> int:
    fig, axes= plt.subplots(nrows = 1, ncols = 4, figsize=(8,2))     
    markevery_dict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1,8:1,9:1,10:1}
    marker_dict={0:'D', 1:'P', 2:'X', 3:'o', 4:'P', 5:'h', 6:6, 7:11,8:10, 9:9,10:7}

    label={0:'lpc',\
           1:'lpnc',\
           2:'onlinebpc',\
           3:'smpc',\
           4:'smpccexact',
           5:'smpcc',
           6:'smpca',
           7:'onlinebpc',\
           8:'smpct_avg', # smpct_avg
           9:'smpct_forecast',
           10:'smpca_avg'
    } 

    plot_label={0:'P-Controller',\
           1:'P-Controller (No-State)',\
           2:'BPC',\
           3:'SMPC',\
           4:'SMPCC-Exact',\
           5:'SMPCC',
           6:'SMPCA',
           7:'CRC',
           8:'SMPCT-Avg',
           9:'SMPCT-Forecast',
           10:'SMPCA-Avg'
           
    }
    y_max_util = []
    #for col in [4,2,5,10,11]:#range(4):
    for col in [1,7]:

        #x = [0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100]
        x = [ 0.01, 0.1, 1., 10., 100., 1000.]
        #for gamma in [.1]:
        #for gamma in ['None']:
        gamma = .1
        #for lr in [0.001, 0.01, 0.1]:
        hyparams = [(lr, init__)
            for lr in [ 0.1]
            for init__ in ['zero', 'one']]
        for (lr, init__) in hyparams:
            y_dcg = []
            y_weighted_objective = []
            y_unsat_mean = []
            y_unsat_sum = []
            y_time = []

            #for c in [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]:
            #for c in ["0.0001_0.001", "0.001_0.001", "0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]:
            for c in ["0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]:
                if label[col] == 'bpc':
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [('None', c, init_, b, bo, gamma) \
                        for b in ['None']
                        for bo in ['None']
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] in ['smpca', 'smpccexact', 'smpcc']:
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [('None', c, init_, b, bo, gamma)\
                        for b in [1]
                        for bo in [10]
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] in ['smpct_avg', 'smpct_forecast', 'smpca_avg']:
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [(lr, c, init_, b, bo, gamma)\
                        for b in [1]
                        for bo in [1]
                        for init_ in ['zero']]
                        #for lr in ['1.0']]
                elif label[col] in ['lpc', 'lpnc']:
                    args.algorithm = label[col] 
                    #gamma = .1
                    params = [('None', c, init_, b, bo, gamma)\
                        for b in ['None']
                        for bo in ['None']
                        for init_ in ['None']]
                        #for lr in ['None']]
                elif label[col] in ['onlinebpc','onlinebpcclip']:
                    args.algorithm = label[col] 
                    gamma = 1
                    params = [(lr, c, init__, b, bo, gamma) \
                        for b in ['None']
                        for bo in ['None']]
                        #for init_ in ['one']]
                        #for lr in ['0.1']]
                else:
                    raise Exception(f'Unknow alg: {label[col]}') 
        
                unsatisfaction_mean = -np.inf
                unsatisfaction_std = np.inf
                dcg = -np.inf
                file = None
        
                for param in params:
                    (lr, c, init_, b, bo, gamma) = param
                    name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                    name = name.replace("/","_")
        
                    try:
                        df = pd.read_pickle(f'tv_audience/{name}.pkl')
                    except:
                        print(f'tv_audience/{name}.pkl')
       
                    if math.isclose(df.iloc[-1]['Unsatisfaction Mean'] , 0) and\
                       math.isclose(df.iloc[-1]['Unsatisfaction Std'] , 0) and \
                       df.iloc[-1]['Sum DCG'] > dcg:
                           dcg =  df.iloc[-1]['Sum DCG']
                           file = f'tv_audience/{name}.pkl'
        
                if file == None:
                    print(f'{args.algorithm} -- unsatisfied')
                    for param in params:
                        (lr, c, init_, b, bo, gamma) = param
                        name = f"{args.algorithm}_" + "_".join([str(x) for x in param ])
                        name = name.replace("/","_")
            
                        df = pd.read_pickle(f'tv_audience/{name}.pkl')
                        if df.iloc[-1]['Unsatisfaction Mean'] > unsatisfaction_mean and\
                           df.iloc[-1]['Unsatisfaction Std'] < unsatisfaction_std: 
                               unsatisfaction_mean = df.iloc[-1]['Unsatisfaction Mean']
                               unsatisfaction_std = df.iloc[-1]['Unsatisfaction Std']
                               file = f'tv_audience/{name}.pkl'
                else:
                    print(f'Best {args.algorithm} file: {file}i {dcg}')
        
                df = pd.read_pickle(file)
     
                y_dcg.append(df['Sum DCG'].iloc[-1])
                y_weighted_objective.append(df['Weighted Objective'].iloc[-1])
                y_unsat_mean.append(df['Unsatisfaction Mean'].iloc[-1])
                y_unsat_sum.append(df['Unsatisfaction Sum'].iloc[-1])
                try:
                    y_time.append(df['elapsed_time'].iloc[-1]/60.)
                except:
                    import pdb; pdb.set_trace()

            #axes[0].set_title(f'{args.algorithm}', fontsize='small')

            # Plot DCG 
            axes[0].plot(x, y_dcg, marker=marker_dict[col], mfc="w", zorder=10, markevery=markevery_dict[col], label=f'{plot_label[col]}_{lr}_{init__}' if col == 7 else f'{plot_label[col]}')
            axes[0].set_title('DCG', fontsize='small')
            axes[0].set_xlabel('c penality', fontsize='small')
            axes[0].set_xscale('log')
            #axes[0].set_xlim(0, 100)
            #axes[0].set_xticks(np.arange(.001, 100, step=10))
    
#            # Plot Unsatisfaction Mean 
#            axes[1].plot(x, y_unsat_mean, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
#            axes[1].set_title('Unsat. Mean', fontsize='small')
#            axes[1].set_xlabel('c penality', fontsize='small')
#            axes[1].set_xscale('log')
    
            # Plot Unsatisfaction Sum 
            axes[1].plot(x, y_unsat_sum, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
            axes[1].set_title('C * Unsat. Sum', fontsize='small')
            axes[1].set_xlabel('c penality', fontsize='small')
            axes[1].set_xscale('log')
            #axes[1].set_ylim(-0.5, 0.5)
    
            # Plot Weighted Objective
            axes[2].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col])
            axes[2].set_title(r'Weighted Objective', fontsize='small')
            axes[2].set_xlabel('c penality', fontsize='small')
            axes[2].set_xscale('log')

            # Elapsed time
            axes[3].plot(x, y_time, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col])
            axes[3].set_title(r'Elapsed Time (minutes)', fontsize='small')
            axes[3].set_xlabel('c penality', fontsize='small')
            axes[3].set_xscale('log')

            #if col != 2 and col != 5:
            #if col == 2 or col == 5 or col == 10 or col == 12 or col ==13 or col == 14 or col == 15 or col == 7 or col ==8:
            if col == 1:
                break

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, ncol=5, frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(f'all_tv.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
