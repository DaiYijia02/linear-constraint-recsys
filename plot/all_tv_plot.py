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
    fig, axes= plt.subplots(nrows = 1, ncols = 3, figsize=(6,2), sharex=True, sharey='row')    
    markevery_dict = {0:1, 1:2, 2:1, 3:2}
    marker_dict={0:'P', 1:'D', 2:'X', 3:'o'}
    label={0:'base', 1:'pc', 2:'olp', 3:'smpc'}

    #max_util = {} 
    y_max_util = []
    for col in [0, 2, 3]:#range(4):
        y_dcg = []
        y_exposure = []
        y_delta = []
#        y_sacrifice_ratio = []
#        y_unsatisfaction_mean = []
#        y_unsatisfaction_std = []
#        y_unsatisfaction_max = []
        y_reward_minus_cost = []
        y_cost = []

        x = [1., 1.5, 2., 2.5, 3, 3.5, 4., 4.5, 5., 5.5]
        for target in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4., 4.5, 5., 5.5]:
            #if args.algorithm == 'olp':
            if col == 2:
                args.algorithm = 'olp'
                params = [(lr, c, init_, target, b, bo) \
                    for b in ['None']
                    for bo in ['None']
                    for init_ in ['zero', 'one']
                    for lr in [10, 5,  1, .1, .01, .001, .0001]
                    for c in ['None']]
            #elif args.algorithm == 'smpca':
            elif col == 3:
                args.algorithm = 'smpca'
                params = [(lr, c, init_, target, b, bo)\
                    for b in [1, 3, 5, 10]
                    for bo in [1, 3, 5, 10]
                    for init_ in ['None']
                    for lr in ['None']
                    for c in [1]]#,5, 10, .0001, .001, .01, .1]]
            #elif args.algorithm == 'base' or args.algorithm == 'pc':
            elif col == 0 or col == 1:
                if col == 0:
                    args.algorithm = 'base'
                elif col == 1:
                    args.algorithm = 'pc'
                params = [(lr, c, init_, target, b, bo)\
                    for b in ['None']
                    for bo in ['None']
                    for init_ in ['None']
                    for lr in ['None']
                    for c in ['None']]
            else:
                raise Exeption('Unknow alg') 
    
            unsatisfaction_mean = -np.inf
            unsatisfaction_std = np.inf
            dcg = -np.inf
            file = None
    
            for param in params:
                (lr, c, init_, target, b, bo) = param
                name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                name = name.replace("/","_")
    
                try:
                    df = pd.read_pickle(f'tv_audience/{name}.pkl')
                except:
                    print(f'tv_audience/{name}.pkl')
   
                if math.isclose(df.iloc[-1]['Unsatisfaction Mean'] , 0) and\
                   math.isclose(df.iloc[-1]['Unsatisfaction Std'] , 0) and \
                   df.iloc[-1]['DCG'] > dcg:
                       dcg =  df.iloc[-1]['DCG']
                       file = f'tv_audience/{name}.pkl'
    
            if file == None:
                print(f'{args.algorithm} -- unsatisfied')
                for param in params:
                    (lr, c, init_, target, b, bo) = param
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
    
            skip = 10
    
            df = pd.read_pickle(file)
            #x = list(df.index)
 
            y_dcg.append(df['DCG'].iloc[-1])
            y_exposure.append(df['state'].iloc[-1])
            y_delta.append(df['delta'].iloc[-1])
#            y_sacrifice_ratio.append(df['Sacrifice Ratio'].iloc[-1])
#            y_unsatisfaction_mean.append(df['Unsatisfaction Mean'].iloc[-1])
#            y_unsatisfaction_std.append(df['Unsatisfaction Std'].iloc[-1])
#            y_unsatisfaction_max.append(df['Unsatisfaction Max'].iloc[-1])
            y_reward_minus_cost.append(df['reward_minus_cost'].iloc[-1])
            y_cost.append(df['Unsatisfaction Sum'].iloc[-1])
             
            if col == 0:
                max_util = df['DCG'].iloc[-1]
                y_max_util.append(df['DCG'].iloc[-1])
   
        #axes[0].set_title(f'{args.algorithm}', fontsize='small')

        # Plot Utility
        #util = df['DCG'].tolist()
        #axes[0].plot(x, y_max_util, color='lightgrey', linestyle='--', label='highest utility', marker='o', zorder=5, mfc="w") 
        axes[0].plot(x, y_dcg, marker=marker_dict[col], mfc="w", zorder=10, markevery=markevery_dict[col], label=label[col])
        #axes[0][col].set_ylim([1.5, 2.5])
        #if col == 0:
        axes[0].set_title('utility', fontsize='small')
        axes[0].set_xlabel('expousre', fontsize='small')
        axes[0].set_ylabel('dcg', fontsize='small')
        #axes[0].set_yticks([0,1,2])

#        # Plot Exposure
#        #delta = np.array(df['delta'].tolist())
#        axes[1][col].plot(x, np.array(y_delta)[:, 0], marker='X', mfc='w', color='lightgrey', linestyle='--', markevery=1)
#        axes[1][col].plot(x, np.array(y_delta)[:, 1], marker='o', mfc='w', color='lightgrey', linestyle='--', markevery=2)
#        axes[1][col].plot(x, np.array(y_delta)[:, 2], marker='D', mfc='w', color='lightgrey', linestyle='--', markevery=2)
#        axes[1][col].plot(x, np.array(y_delta)[:, 3], marker='P', mfc='w', color='lightgrey', linestyle='--', markevery=2)
#        if col == 0:
#            axes[1][col].set_ylabel('exposure', fontsize='small')
#
#        #exposure = np.array(df['state'].tolist())
#        axes[1][col].plot(x, np.array(y_exposure)[:, 0], marker='X', mfc="w", markevery=1)
#        axes[1][col].plot(x, np.array(y_exposure)[:, 1], marker='o', mfc="w", markevery=2)
#        axes[1][col].plot(x, np.array(y_exposure)[:, 2], marker='D', mfc="w", markevery=2)
#        axes[1][col].plot(x, np.array(y_exposure)[:, 3], marker='P', mfc="w", markevery=2)
    
#        # Sacrifice Ratio
#        axes[2][col].plot(x, y_sacrifice_ratio)
#        if col == 0:
#            axes[2][col].set_ylabel('sacrifice ratio ($\leftarrow$)', fontsize='small')
#
#        # Unsatisfaction Mean
#        axes[3][col].plot(x, y_unsatisfaction_mean)
#        if col == 0:
#            axes[3][col].set_ylabel('uns mean', fontsize='small')
#
#        # Unsatisfaction Std
#        axes[4][col].plot(x, y_unsatisfaction_std)
#        if col == 0:
#            axes[4][col].set_ylabel('uns std', fontsize='small')

        # Plot Cost
        axes[1].plot(x, y_cost, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
        #if col == 0:
        axes[1].set_title('cost', fontsize='small')
        axes[1].set_ylabel('1/rank', fontsize='small')
        axes[1].set_xlabel('exposure', fontsize='small')

        # Plot Reward - Cost
        #axes[5][col].plot(x, y_unsatisfaction_max)
        axes[2].plot(x, y_reward_minus_cost, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col])
        #if col == 0:
        axes[2].set_title(r'utility - cost', fontsize='small')
        axes[2].set_xlabel('exposure', fontsize='small')
     

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, ncol=4, frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(f'all_tv.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
