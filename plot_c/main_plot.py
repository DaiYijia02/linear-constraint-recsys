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
parser.add_argument('--nhrs', type=int, default=2)
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

def main() -> int:
    fig, axes= plt.subplots(nrows = 3, ncols = 3, figsize=(4,2), gridspec_kw = {'hspace':0.2, 'wspace':0.6}, sharex=True)     

    markevery_dict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1,8:1,9:1,10:1, 11:1}
    marker_dict={0:'D', 1:'P', 2:'s', 3:'o', 4:'P', 5:'h', 6:6, 7:11,8:10, 9:9,10:7,11:'D'}

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
                2:r'bpc',\
                3:r'cr',\
                4:r'CA-Hinge',\
                5:r'cp',\
                6:r'CP-Hinge',\
                7:r'cp',\
                8:r'CPB-Hinge'
    }
    init_ca = ['zero']

    y_max_util = []
    for col in [0, 2, 3, 7]:
        x = [0.01, 0.1, 1., 10., 100.]
        hyparams = [(hinge_min, task)
            for hinge_min in [0.0, ]
            for task in ['kuai', 'zf_tv', 'movie_len_top_10']]#, 'early_and_late']]
        for (hinge_min, task) in hyparams:
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
                b_sweep = [20,50]
            elif task == 'early_and_late':
                r_type = 'online_relevance'
                lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
                b_sweep = [1]
            elif task == 'late':
                r_type = 'online_relevance'
                lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
                b_sweep = [1]
                cs = ["0.01", "0.1", "1.", "10.", "100."]
                init_ca = ['zero']
            elif task == 'early':
                r_type = 'online_relevance'
                lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
                b_sweep = [1]
                cs = ["0.01", "0.1", "1.", "10.", "100."]
                init_ca = ['zero','one']
            elif task == 'tv_audience':
                r_type = 'sequence_relevance'
                lr_sweep = [10.0, 1.0, 0.1, 0.01]
                b_sweep = [20, 50]
            elif task == 'zf_tv':
                r_type = 'online_relevance'
                lr_sweep = [10.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
                b_sweep = [20, 40, 50]
                cs = ["0.01", "0.1", "1.", "10.", "100."]
            elif task == 'movie_len_top_10':
                r_type = 'online_relevance'
                lr_sweep = [10.0, 1.0, 0.1, 0.01]
                b_sweep = [20, 50]

            for c in cs:
            #for c in ["0.01", "0.1", "1.", "10.", "100."]:
                if label[col] == 'bpc':
                    args.algorithm = label[col] 
                    params = [('0.0001', c, init_, b, bo, r_type, 'None', 'None', 'None') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                elif label[col] == 'bpcphi' or label[col] == 'bpcphit':
                    args.algorithm = label[col] 
                    params = [('0.0001', c, init_, b, bo, r_type, hinge_min, 'None', 'None') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                elif label[col] == 'oracle':
                    args.algorithm = label[col] 
                    params = [('0.0001', c, init_, b, bo, r_type, 'None', 'None', 'None') \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['None']]
                elif label[col] in ['cpb']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, 'offline_relevance', 'None', beta, eps)\
                        for b in b_sweep 
                        for bo in b_sweep 
                        for init_ in ['zero']
                        for beta in [0.5]#, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                elif label[col] in ['cpbhinge']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, 'offline_relevance', hinge_min, beta, eps)\
                        for b in b_sweep 
                        for bo in b_sweep 
                        for init_ in ['zero']
                        for beta in [0.5, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                elif label[col] in ['cp']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, 'offline_relevance', 'None', beta, eps)\
                        for b in b_sweep 
                        for bo in [1]
                        for init_ in ['zero']
                        for beta in [0.5, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                elif label[col] in ['cphinge']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, 'offline_relevance', hinge_min, beta, eps)\
                        for b in b_sweep 
                        for bo in [1]
                        for init_ in ['zero']
                        for beta in [0.5, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                elif label[col] in ['ca']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, r_type, 'None', beta, eps) \
                        for b in [1]
                        for bo in [1]
                        for init_ in init_ca #['zero']
                        for beta in [0.5, 0.9, 0.98]
                        for eps in ['1e-05', '1e-08']
                        for lr in lr_sweep]
                elif label[col] in ['cahinge']:
                    args.algorithm = label[col] 
                    params = [(lr, c, init_, b, bo, r_type, hinge_min, beta, eps) \
                        for b in [1]
                        for bo in [1]
                        for init_ in ['zero']
                        for beta in [0.5, 0.9, 0.98]
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

            markersize = 3 
            # Plot DCG 
            if task == 'kuai':
                ax_idx = 0
                axes[0][0].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], label=f'{plot_label[col]}' if col in [2,4, 6] else f'{plot_label[col]}', markersize=markersize)
                axes[0][0].set_title(r'kuairec', fontsize='xx-small')
    
            # Plot Unsatisfaction Sum 
            elif task == 'zf_tv':
                ax_idx = 1
                axes[0][1].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize)
                axes[0][1].set_title(r'tv audience', fontsize='xx-small')
    
            # Plot Weighted Objective
            elif task == 'movie_len_top_10':
                ax_idx = 2
                axes[0][2].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize)
                axes[0][2].set_title(r'last.fm', fontsize='xx-small')

            elif task == 'early_and_late':
                ax_idx = 3
                axes[0][3].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize)
                axes[0][3].set_title(r'early and late', fontsize='xx-small')

            if ax_idx == 0:
                axes[2][ax_idx].set_ylabel('objective', fontsize='xx-small')
            axes[2][ax_idx].plot(x, y_weighted_objective, marker=marker_dict[col], mfc="w", markevery=markevery_dict[col], markersize=markersize)
            axes[2][ax_idx].set_xscale('log')
            axes[2][ax_idx].tick_params(axis='both', which='major', labelsize='xx-small')
            axes[2][ax_idx].tick_params(axis='both', which='minor', labelsize='xx-small')
            axes[2][ax_idx].set_xlabel(r'cost vector $\phi$', fontsize='xx-small')

            # Plot Unsatisfaction Sum
            if ax_idx == 0:
                axes[0][ax_idx].set_ylabel('dcg', fontsize='xx-small')
            #axes[0][ax_idx].plot(x, y_dcg, marker=marker_dict[col], mfc="w", zorder=10, markevery=markevery_dict[col], markersize=markersize)
            axes[0][ax_idx].set_xscale('log')
            axes[0][ax_idx].tick_params(axis='both', which='major', labelsize='xx-small')
            axes[0][ax_idx].tick_params(axis='both', which='minor', labelsize='xx-small')

            # Plot Weighted Objective
            if ax_idx == 0:
                axes[1][ax_idx].set_ylabel('violation cost', fontsize='xx-small')
            axes[1][ax_idx].plot(x, y_unsat_sum, marker=marker_dict[col], mfc='w', markevery=markevery_dict[col], markersize=markersize)
            #axes[2].set_title(r'Weighted Objective', fontsize='small')
            axes[1][ax_idx].set_xscale('log')
            axes[1][ax_idx].tick_params(axis='both', which='major', labelsize='xx-small')
            axes[1][ax_idx].tick_params(axis='both', which='minor', labelsize='xx-small')

            #if col in [3,4,5,6]:
            #if col in [0, 1, 3, 5]:
            #    break

    for j in range(3):
       for i in range(3):
           #axes[j, i].autoscale(enable=True, axis='both', tight=True)
           axes[j, i].margins(.15)
           #axes[j, i].set_xlim([0.001, 500.0])
           axes[j, i].set_xticks([.01, 1, 100])
           #axes[j, i].set_xticklabels([.01, 1, 100])
           #axes[j, i].minorticks_off()

           #ymin, ymax = axes[j, i].get_ylim()
           #axes[j, i].set_ylim([ymin-3, ymax+3])

    labelx = -0.5
    for j in range(3):
        axes[j, 0].yaxis.set_label_coords(labelx, 0.5)

    lines_labels = [ax.get_legend_handles_labels() for idx, ax in enumerate(fig.axes)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, ncol=4, frameon=False, fontsize="xx-small")

    fig.tight_layout()
    fig.savefig(f'main_plot.pdf', format='pdf',  bbox_inches = 'tight', pad_inches = 0)
    return 0
    
if __name__ == '__main__':
      sys.exit(main())  # next section explains the use of sys.exit
