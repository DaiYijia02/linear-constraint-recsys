import os
from datetime import datetime
import argparse
import time
import socket
import uuid

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

_folder = f'{args.base_save_dir}/results'
if not os.path.exists(_folder):
    os.makedirs(_folder)

jobs = []
bs = [1, 5, 10, 15, 20,25,50] if args.algorithm in ['smpca', 'smpcb', 'smpcc', 'smpct_avg', 'smpct_forecast', 'smpca_avg'] else ['None']
bos = [1, 5, 10, 15, 20, 25,50] if args.algorithm in ['smpca', 'smpcb', 'smpcc', 'smpct_avg', 'smpct_forecast', 'smpca_avg'] else ['None']
lrs = [0.01, 0.1, 1.] if args.algorithm in ['onlinebpc', 'smpct_avg', 'smpct_forecast','smpca_avg'] else ['None']
gammas = [1] if args.algorithm in ['smpca', 'smpcb', 'smpcc', 'bpc', 'onlinebpc', 'smpct_avg', 'smpct_forecast', 'smpca_avg'] else [10, 1, 0.1, .01]
inits_ = ['zero'] if args.algorithm in ['onlinebpc', 'smpct_avg', 'smpct_forecast', 'smpca_avg'] else ['None']
#cs = ["0.0001_0.0001", "0.001_0.001", "0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]
cs = ["0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]


assert (args.algorithm in ['olp', 'smpca', 'smpcb', 'smpcc', 'smpcd', 'base', 'pc', 'bpc', 'lpc', "lpnc", 'onlinebpc', 'smpct_avg', 'smpct_forecast', 'smpca_avg'])

params = [(lr, c, init_, b, bo, gamma)\
    for lr in lrs
    for c in cs
    for init_ in inits_
    for b in bs
    for bo in bos
    for gamma in gammas
]

for param in params:
    (lr, c, init_, b, bo, gamma) = param
    name = f"{args.algorithm}_" + "_".join([str(x) for x in param ])
    name = name.replace("/","_")

    cmd = f'python -u -m simulate experiments/multi_group.yml kuai  {args.algorithm} --output_dir outputs/{args.algorithm} --c {" ".join(c.split("_"))} --metrics_file_name {name} --gamma {gamma} '
    if args.algorithm in ['smpca', 'smpcb', 'smpcc', 'smpct_avg', 'smpct_forecast', 'smpca_avg']:
        cmd += f'--b {b} --bo {bo} '
    if args.algorithm in ['onlinebpc', 'smpct_avg', 'smpct_forecast', 'smpca_avg']:
        cmd += f'--bpc_lr {lr} --bpc_init {init_} '
    jobs.append((cmd, name, param))

output_dir = os.path.join(args.base_save_dir, args.output_dirname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

id_name = uuid.uuid4()
now_name = f'{args.base_save_dir}/output/now_{id_name}.txt'
was_name = f'{args.base_save_dir}/output/was_{id_name}.txt'
log_name = f'{args.base_save_dir}/output/log_{id_name}.txt'
err_name = f'{args.base_save_dir}/output/err_{id_name}.txt'
num_commands = 0
jobs = iter(jobs)
done = False
threshold = 999

#for (game, agent_name, game_type, seed, lr, demo) in params:
while not done:
    #for (game, agent_name, game_type, seed, demo, learning_rate, data_observation_norm, absorbing_state) in params:
    for (cmd, name, params) in jobs:

        if os.path.exists(now_name):
            file_logic = 'a'  # append if already exists
        else:
            file_logic = 'w'  # make a new file if not
            print(f'creating new file: {now_name}')

        with open(now_name, 'a') as nowfile,\
             open(was_name, 'a') as wasfile,\
             open(log_name, 'a') as output_namefile,\
             open(err_name, 'a') as error_namefile:

            if nowfile.tell() == 0:
                print(f'a new file or the file was empty: {now_name}')

            now = datetime.now()
            datetimestr = now.strftime("%m%d_%H%M:%S.%f")

            num_commands += 1
            nowfile.write(f'{cmd}\n')
            wasfile.write(f'{cmd}\n')

            output_dir = os.path.join(args.base_save_dir, args.output_dirname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
            error_namefile.write(f'{(os.path.join(output_dir, name))}.error\n')
            if num_commands == threshold:
                break
    if num_commands != threshold:
        done = True

    # Make a {name}.slurm file in the {output_dir} which defines this job.
    #slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    start=1
    slurm_script_path = os.path.join(output_dir, f'submit_{start}_{num_commands}.slurm')
    slurm_command = "sbatch %s" % slurm_script_path


    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --array=1-{num_commands}\n")
        slurmfile.write("#SBATCH --output=/dev/null\n")
        slurmfile.write("#SBATCH --error=/dev/null\n")
        slurmfile.write("#SBATCH --requeue\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        #slurmfile.write("#SBATCH --account=scavenger\n")
        slurmfile.write("#SBATCH --partition default_partition\n")
        slurmfile.write("#SBATCH -c 8\n")
        slurmfile.write("#SBATCH --mem=15G\n")
        slurmfile.write("#SBATCH --exclude=g2-cpu-01,g2-cpu-02,g2-cpu-03,g2-cpu-04,g2-cpu-05,g2-cpu-06,g2-cpu-07,g2-cpu-08,g2-cpu-09,g2-cpu-10,g2-cpu-11,g2-cpu-26,g2-cpu-27,g2-cpu-28,g2-cpu-29,g2-cpu-30,g2-cpu-97,g2-cpu-98,g2-cpu-99\n")

        slurmfile.write("\n")
        slurmfile.write("cd " + args.base_save_dir + '\n')
        slurmfile.write("conda activate rank38\n")
        slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {log_name} | tail -n 1) --error=$(head -n    $SLURM_ARRAY_TASK_ID {err_name} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_name} | tail -n 1)\n" )
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    num_commands = 0
    id_name = uuid.uuid4()
    now_name = f'{args.base_save_dir}/output/now_{id_name}.txt'
    was_name = f'{args.base_save_dir}/output/was_{id_name}.txt'
    log_name = f'{args.base_save_dir}/output/log_{id_name}.txt'
    err_name = f'{args.base_save_dir}/output/err_{id_name}.txt'
