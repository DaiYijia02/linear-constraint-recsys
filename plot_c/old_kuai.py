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
    fig, axes= plt.subplots(nrows = 1, ncols = 3, figsize=(8,2))     
    #markevery_dict = {0:1, 1:1, 2:1, 3:1, 4:1,}
    marker_dict={0:'D', 1:'P', 2:'X', 3:'o', 4:'1', 5:'h', 6:6, 7:7, 8:'P', 9:9, 10:10}
    label={0:'LP-p-controller', 
           1:'lpnc', 
           2:'onlinebpc', 
           3:'smpc', 
           4:'LP-No-State-p-controller',
           5:'bpcapprox',
           6:'bpapproxsum',
           7:'bpapproxinside',
           8:'bpapproxoutside',
           9:'lpcapprox',
           10: 'lpc'}

    plot_label={0:'P-Controller',\
           1:'CO',\
           2:'CA',\
           3:'SMPC',\
           4:'SMPCC-Exact',\
           5:'SMPCC',
           6:'SMPCA',
           7:'CRC',
           8:'SMPCT-Avg',
           9:'SMPCT-Forecast',
           10:'SMPCA-Avg'
    }

    for col in [1,2]:#range(4):
        #x = [0.001, 0.01, 0.1, 1., 10., 100.]
        #gamma = 0.1 
        c = "1._1." 

        if label[col] == 'onlinebpc':
            args.algorithm = 'onlinebpc'
            gamma = 1 
            params = [(lr, c, init_, b, bo, gamma) \
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'bpcapprox':
            args.algorithm = 'bpcapprox'
            gamma = 1 
            params = [(lr, c, init_, b, bo, gamma) \
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'bpapproxsum':
            args.algorithm = 'bpapproxsum'
            gamma = 1 
            params = [(lr, c, init_, b, bo, gamma) \
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'bpapproxinside':
            args.algorithm = 'bpapproxinside'
            gamma = 1 
            params = [(lr, c, init_, b, bo, gamma) \
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'bpapproxoutside':
            args.algorithm = 'bpapproxoutside'
            gamma = 1 
            params = [(lr, c, init_, b, bo, gamma) \
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'smpca':
            args.algorithm = 'smpca'
            #gamma = .1
            params = [(lr, c, init_, b, bo, gamma)\
                for b in [100]
                for bo in [10]
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'pc':
            args.algorithm = 'pc'
            #gamma = .1
            params = [(lr, c, init_, b, bo, .1)\
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'lpc':
            args.algorithm = 'lpc'
            gamma = .1
            params = [(lr, c, init_, b, bo, .1)\
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'lpcapprox':
            args.algorithm = 'lpcapprox'
            gamma = .1
            params = [(lr, c, init_, b, bo, .1)\
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        elif label[col] == 'lpnc':
            args.algorithm = 'lpnc'
            gamma = .1
            params = [(lr, c, init_, b, bo, .1)\
                for b in ['None']
                for bo in ['None']
                for init_ in ['None']
                for lr in ['None']]
        else:
            raise Exception('Unknow alg') 

        unsatisfaction_mean = -np.inf
        unsatisfaction_std = np.inf
        dcg = -np.inf
        file = None

        for param in params:
            (lr, c, init_, b, bo, gamma) = param
            name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
            name = name.replace("/","_")

            try:
                df = pd.read_pickle(f'kuai/{name}.pkl')
            except:
                print(f'kuai/{name}.pkl')
   
            if math.isclose(df.iloc[-1]['Unsatisfaction Mean'] , 0) and\
               math.isclose(df.iloc[-1]['Unsatisfaction Std'] , 0) and \
               df.iloc[-1]['Sum DCG'] > dcg:
                   dcg =  df.iloc[-1]['Sum DCG']
                   file = f'kuai/{name}.pkl'

        if file == None:
            print(f'{args.algorithm} -- unsatisfied')
            for param in params:
                (lr, c, init_, b, bo, gamma) = param
                name = f"{args.algorithm}_" + "_".join([str(x) for x in param ])
                name = name.replace("/","_")
    
                df = pd.read_pickle(f'kuai/{name}.pkl')
                if df.iloc[-1]['Unsatisfaction Mean'] > unsatisfaction_mean and\
                   df.iloc[-1]['Unsatisfaction Std'] < unsatisfaction_std: 
                       unsatisfaction_mean = df.iloc[-1]['Unsatisfaction Mean']
                       unsatisfaction_std = df.iloc[-1]['Unsatisfaction Std']
                       file = f'kuai/{name}.pkl'
        else:
            print(f'Best {args.algorithm} file: {file}i {dcg}')

        df = pd.read_pickle(file)
 
#        y_dcg.append(df['Sum DCG'].iloc[-1])
#        y_exposure.append(df['state'].iloc[-1])
#        y_delta.append(df['delta'].iloc[-1])
#        y_weighted_objective.append(df['Weighted Objective'].iloc[-1])
#        #y_cost.append(df['C * Unsatisfaction Sum'].iloc[-1])
#        y_unsat_mean.append(df['Unsatisfaction Mean'].iloc[-1])
#        y_unsat_sum.append(df['Unsatisfaction Sum'].iloc[-1])
#
#        y_status.append( np.mean(df['status'].iloc[-1]) )
        y_dcg = df['Sum DCG'].tolist()
        y_unsat_sum = df['Unsatisfaction Sum'].tolist()
        y_weighted_objective = df['Weighted Objective'].tolist()
        x = range(len( df['Sum DCG'].tolist() ))
         
        #axes[0].set_title(f'{args.algorithm}', fontsize='small')

        # Plot DCG 
        axes[0].plot(x, y_dcg, mfc="w", zorder=10, label=f'{label[col]}_{gamma}')
        axes[0].set_title('DCG', fontsize='small')
        axes[0].set_xlabel('steps', fontsize='small')
        axes[0].set_yscale('log')
        axes[0].set_xscale('log')
        #axes[0].set_ylim(1990, 2020)
        #axes[0].set_xscale('log')
        #axes[0].set_xlim(0, 100)
        #axes[0].set_xticks(np.arange(.001, 100, step=10))

#            # Plot Unsatisfaction Mean 
#            axes[1].plot(x, y_unsat_mean, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
#            axes[1].set_title('Unsat. Mean', fontsize='small')
#            axes[1].set_xlabel('time steps', fontsize='small')
#            axes[1].set_xscale('log')

        # Plot Unsatisfaction Sum 
        axes[1].plot(x, y_unsat_sum, mfc='w')
        axes[1].set_title('C * Unsat. Sum', fontsize='small')
        axes[1].set_xlabel('time steps', fontsize='small')
        #axes[1].set_xscale('log')
        #axes[1].set_yscale('log')

        axes[1].set_ylim(0, 100)

        # Plot Weighted Objective
        axes[2].plot(x, y_weighted_objective, mfc="w")
        axes[2].set_title(r'Weighted Objective', fontsize='small')
        axes[2].set_xlabel('time steps', fontsize='small')
        #axes[2].set_ylim(1800, 2020)
        axes[2].set_yscale('log')
        axes[2].set_xscale('log')

        #axes[2].set_xscale('log')

        # Plot Weighted Objective
#            axes[4].plot(x, y_status, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col])
#            axes[4].set_title(r'Status', fontsize='small')
#            axes[4].set_xlabel('c penality', fontsize='small')
#            axes[4].set_xscale('log')

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, ncol=4, frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(f'single_plot.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
