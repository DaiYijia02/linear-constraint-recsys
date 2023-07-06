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
parser.add_argument('--stat', choices=['forecast', 'samples'], required=True)
args = parser.parse_args()

def main() -> int:
    fig, axes= plt.subplots(nrows = 1, ncols = 2, figsize=(3., 1.), gridspec_kw = {'wspace':0.5} )     
    markevery_dict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1,8:1,9:1,10:1, 11:1}
    marker_dict={0:'D', 1:'P', 2:'X', 3:'o', 4:'P', 5:'h', 6:6, 7:11,8:10, 9:9,10:7,11:'D'}

    label={0:'oracle',\
           #1:'bpc',\
           2:'bpcphi',\
           3:'ca',\
           4:"cahinge",\
           5:"cp",\
           6:"cphinge",\
           7:'cpb',\
           8:'cpbhinge'
    } 

    plot_label={0:r'oracle',\
                #1:r'BPC',\
                2:r'BPC',\
                3:r'CA',\
                4:r'CA-Hinge',\
                5:r'cp',\
                6:r'CP-Hinge',\
                7:r'cp',\
                8:r'CPB-Hinge'
    }
    init_ca = ['zero']

    y_max_util = []
    for col in [7]:
        x = [0.01, 0.1, 1., 10., 100.]
        hyparams = [(hinge_min, task, b, bo)
            for hinge_min in [0.0]
            for task in ['kuai', 'movie_len_top_10']
            for bo in [1, 5, 20, 50]
            for b in [1, 5, 20, 50]]
        for (hinge_min, task, b, bo) in hyparams:
            y_dcg = []
            y_weighted_objective = []
            y_unsat_mean = []
            y_unsat_sum = []
            y_time = []
            y_delta = []
            y_eposure = []

            cs = ["0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100."]
            if task == 'kuai':
                r_type = 'online_relevance'
                lr_sweep = [0.1, 0.01, 0.001, 0.0001]
                #b_sweep = [50]
                #b_sweep = [20,50]
            elif task == 'movie_len_top_10':
                r_type = 'online_relevance'
                lr_sweep = [10.0, 1.0, 0.1, 0.01]
                cs = ["0.1_0.1", "1._1.", "10._10.", "100._100.", "1000._1000."]
                #b_sweep = [20, 50]

            for c in cs:
                if label[col] in ['cpb']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, 'offline_relevance', 'None', beta, eps)\
                        #for b in b_sweep 
                        #for bo in b_sweep 
                        for init_ in ['zero']
                        for beta in [0.5]#, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                else:
                    raise Exception(f'Unknow alg: {label[col]}') 
        
                # unsatisfaction_mean = -np.inf
                # unsatisfaction_std = np.inf
                # dcg = -np.inf
                weighted_objective = -np.inf
                file = None
        
                for param in params:
                    (lr, c, init_, b, bo, r_type, hinge_min, beta, eps) = param
                    name = f"{args.algorithm}_" + "_".join([str(x) for x in param])
                    name = name.replace("/","_")
        
                    try:
                        df = pd.read_pickle(f'results/{task}/{name}.pkl')
                        #df = pd.read_pickle(f'results/{task}/{name}_dev.pkl')
                    except:
                        print(f'[Error]: file results/{task}/{name}.pkl not found.')
                        #pass
       
                    # if math.isclose(df.iloc[-1]['Unsatisfaction Mean'] , 0) and\
                    #    math.isclose(df.iloc[-1]['Unsatisfaction Std'] , 0) and \
                    #    df.iloc[-1]['Sum DCG'] > dcg:
                    if df.iloc[-1]['Weighted Objective'] > weighted_objective:
                        #    dcg =  df.iloc[-1]['Sum DCG']
                           weighted_objective = df.iloc[-1]['Weighted Objective']
                           file = f'results/{task}/{name}.pkl'
        
                if file == None:
                    print(f'{args.algorithm} -- unsatisfied')
                    for param in params:
                        (lr, c, init_, b, bo, r_type) = param
                        name = f"{args.algorithm}_" + "_".join([str(x) for x in param ])
                        name = name.replace("/","_")
            
                        df = pd.read_pickle(f'results/{task}/{name}.pkl')
                        if df.iloc[-1]['Unsatisfaction Mean'] > unsatisfaction_mean and\
                           df.iloc[-1]['Unsatisfaction Std'] < unsatisfaction_std: 
                               unsatisfaction_mean = df.iloc[-1]['Unsatisfaction Mean']
                               unsatisfaction_std = df.iloc[-1]['Unsatisfaction Std']
                               file = f'results/{task}/{name}.pkl'
                else:
                    print(f'Best {args.algorithm} file: {file.split("/")[-1]} {weighted_objective}')
                    # print(f'Best {args.algorithm} file: {file}i {dcg}')
        
                df = pd.read_pickle(file)
     
                y_dcg.append(df['Sum DCG'].iloc[-1])
                y_weighted_objective.append(df['Weighted Objective'].iloc[-1])
                y_unsat_mean.append(df['Unsatisfaction Mean'].iloc[-1])
                y_unsat_sum.append(df['Unsatisfaction Sum'].iloc[-1])
                try:
                    y_time.append(df['elapsed_time'].iloc[-1]/60.)
                except:
                    import pdb; pdb.set_trace()

#            # Plot DCG 
#            axes[0].plot(x, y_dcg, marker=marker_dict[col], mfc="w", zorder=10, markevery=markevery_dict[col], label=f'{plot_label[col]}' if col in [2,4, 6] else f'{plot_label[col]}')
#            axes[0].set_title('DCG', fontsize='xx-small')
#            axes[0].set_xlabel(r'cost vector $\phi$', fontsize='xx-small')
#            axes[0].set_xscale('log')
#    
#            # Plot Unsatisfaction Sum 
#            axes[1].plot(x, y_unsat_sum, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col])
#            axes[1].set_title(r'Violation Cost', fontsize='xx-small')
#            axes[1].set_xlabel(r'cost vector $\phi$', fontsize='xx-small')
#            axes[1].set_xscale('log')
            #axes[1].set_ylim(-0.5, 0.5)
    
            # Plot Weighted Objective
            markersize =5 
            if task == 'kuai':
                axes[0].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize, label=f'{b}_{bo}' )
                axes[0].set_title(r'kuairec', fontsize='xx-small')
                axes[0].set_xlabel(r'cost vector $\phi$', fontsize='xx-small')
                axes[0].set_xscale('log')
                axes[0].tick_params(axis='both', which='major', labelsize='xx-small')
                axes[0].tick_params(axis='both', which='minor', labelsize='xx-small')
                axes[0].set_ylabel('objective', fontsize='xx-small')

            # Plot Weighted Objective
            if task == 'movie_len_top_10':
                axes[1].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize)
                axes[1].set_title(r'last.fm', fontsize='xx-small')
                axes[1].set_xlabel(r'cost vector $\phi$', fontsize='xx-small')
                axes[1].set_xscale('log')
                axes[1].tick_params(axis='both', which='major', labelsize='xx-small')
                axes[1].tick_params(axis='both', which='minor', labelsize='xx-small')

            for j in range(2):
                 axes[j].margins(.15)
                 axes[j].set_xticks([.01, 1, 100])
            #if col in [3,4,5,6]:
            #if col in [0, 1, 3, 5]:
            #    break

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=False, ncol=4, frameon=False, fontsize="x-small")

    fig.tight_layout()
    fig.savefig(f'stats_{args.stat}.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
