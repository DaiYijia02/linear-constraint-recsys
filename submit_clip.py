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
parser.add_argument('--task', type=str)
args = parser.parse_args()

_folder = f'{args.base_save_dir}/results'
if not os.path.exists(_folder):
    os.makedirs(_folder)
jobs = []
#bs = [1,5,10,20,40,50,60] if args.algorithm in ['cp', 'cphinge'] else [1]
bs = [20, 50] if args.algorithm in ['cpb', 'cpbhinge'] else [1]
bos = [20, 50] if args.algorithm in ['cpb', 'cpbhinge'] else [1]
hinge_mins = [0.0] if args.algorithm in ['cphinge', 'cahinge', 'bpcphi', 'bpcphit', 'cpbhinge'] else ['None']
r_types = ['offline_relevance'] if args.algorithm in ['cp', 'cphinge', 'cpb', 'cpbhinge'] else ['online_relevance']
inits_ = ['zero'] if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge'] else ['None']
#cs = ["0.0001_0.0001", "0.001_0.001", "0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100.","1000._1000."]
cs = ["0.01_0.01", "0.1_0.1", "1._1.", "10._10.", "100._100."]
betas = [0.5, 0.9, 0.98] if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge'] else ['None']
epss = ['1e-05', '1e-08'] if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge'] else ['None']

# Increase the learing rate sweep for certain task
if args.task in ['kuai']:
    lr_sweep = [0.1, 0.01, 0.001, 0.0001]
    shuffle_bootstraps = 'true'
    bs = [20,40,50] if args.algorithm in ['cp', 'cphinge'] else [1]
    #bs = [40,60] if args.algorithm in ['cp', 'cphinge'] else [1]
elif args.task in ['early_and_late']:
    lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
    shuffle_bootstraps = 'false'
    bs = [1]
    bos = [1]
elif args.task in ['tv_audience']:
    lr_sweep = [100.0, 10.0, 1.0, 0.1, 0.01]
    shuffle_bootstraps = 'false'
    bs = [20,40,50] if args.algorithm in ['cp', 'cphinge', 'cpb', 'cpbhinge'] else [1]
    bos = [20,40,50] if args.algorithm in ['cpb', 'cpbhinge'] else [1]
    shuffle_bootstraps = 'false'
elif args.task in ['movie_len_top_10']:
    lr_sweep = [10.0, 1.0, 0.1, 0.01]
    shuffle_bootstraps = 'false'
    bs = [1, 5, 20, 50] if args.algorithm in ['cp', 'cphinge', 'cpb', 'cpbhinge'] else [1]
    bos = [1, 5, 20, 50] if args.algorithm in ['cpb', 'cpbhinge'] else [1]
    cs = ["0.1_0.1", "1._1.", "10._10.", "100._100.", "1000._1000."]
else:
    raise Exception ('Task experimental yml file not set')

lrs = lr_sweep if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge'] else [0.0001]

assert (args.algorithm in ['base', 'bpc', 'bpcphi', 'bpcphit', 'oracle', 'ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge'])

params = [(lr, c, init_, b, bo, r_type, hinge_min, beta, eps, dev)\
    for lr in lrs
    for c in cs
    for init_ in inits_
    for b in bs
    for bo in bos
    for r_type in r_types
    for hinge_min in hinge_mins
    for beta in betas
    for eps in epss
    for dev in ["--dev", ""]
]


for param in params:
    (lr, c, init_, b, bo, r_type, hinge_min, beta, eps, dev) = param

    name = f"{args.algorithm}_" + "_".join([str(x) for x in param ][:-1])
    if dev:
        name += "_dev"
    name = name.replace("/","_").replace("--", "")

    if args.task in ['kuai']:
        experitment_yml = 'multi_group.yml'
    elif args.task in ['early_and_late']:
        experitment_yml = 'multi_early_and_late_group.yml'
    elif args.task in ['tv_audience']:
        experitment_yml = 'multi_tv_audience_group.yml'
    elif args.task in ['movie_len_top_10']:
        experitment_yml = 'multi_movie_len_top_10.yml'
    else:
        raise Exception ('Task experimental yml file not set')


    cmd = f'python -u -m simulate experiments/{experitment_yml} {args.task}  {args.algorithm} --output_dir outputs/{args.algorithm} --c {" ".join(c.split("_"))} --metrics_file_name {name} {dev} '
    if args.algorithm in ['cphinge', 'cahinge', 'bpcphi', 'bpcphit', 'cpbhinge']:
        cmd += f' --hinge_min {hinge_min} '
    if args.algorithm in ['cp', 'cphinge', 'cpb', 'cpbhinge']:
        cmd += f'--b {b} --bo {bo} --relevance_type {r_type} '
    if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge']:
        cmd += f'--bpc_lr {lr} --bpc_init {init_} '
    if args.algorithm in ['ca', 'cahinge', 'cp', 'cphinge', 'cpb', 'cpbhinge']:
        cmd += f' --beta {beta} --eps {eps} '
    cmd += f'--shuffle_bootstraps {shuffle_bootstraps} '
    jobs.append((cmd, name, param))

output_dir = os.path.join(os.path.join(args.base_save_dir, args.output_dirname), args.task)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

id_name = uuid.uuid4()
now_name = f'{args.base_save_dir}/output/{args.task}/now_{id_name}.txt'
was_name = f'{args.base_save_dir}/output/{args.task}/was_{id_name}.txt'
log_name = f'{args.base_save_dir}/output/{args.task}/log_{id_name}.txt'
err_name = f'{args.base_save_dir}/output/{args.task}/err_{id_name}.txt'
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

            output_dir = os.path.join(os.path.join(args.base_save_dir, args.output_dirname), args.task)
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
        #slurmfile.write("#SBATCH --partition scavenger\n")
        slurmfile.write("#SBATCH --partition default_partition\n")
        slurmfile.write("#SBATCH -c 8\n")
        slurmfile.write("#SBATCH --mem=15G\n")
        slurmfile.write("#SBATCH --exclude=g2-cpu-01,g2-cpu-02,g2-cpu-03,g2-cpu-04,g2-cpu-05,g2-cpu-06,g2-cpu-07,g2-cpu-08,g2-cpu-09,g2-cpu-10,g2-cpu-11,g2-cpu-26,g2-cpu-27,g2-cpu-28,g2-cpu-29,g2-cpu-30,g2-cpu-97,g2-cpu-98,g2-cpu-99,g2-compute-96,g2-compute-97\n")

        slurmfile.write("\n")
        slurmfile.write("cd " + args.base_save_dir + '\n')
        slurmfile.write("conda activate rank38\n")
        slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {log_name} | tail -n 1) --error=$(head -n    $SLURM_ARRAY_TASK_ID {err_name} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_name} | tail -n 1)\n" )
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    num_commands = 0
    id_name = uuid.uuid4()
    now_name = f'{args.base_save_dir}/output/{args.task}/now_{id_name}.txt'
    was_name = f'{args.base_save_dir}/output/{args.task}/was_{id_name}.txt'
    log_name = f'{args.base_save_dir}/output/{args.task}/log_{id_name}.txt'
    err_name = f'{args.base_save_dir}/output/{args.task}/err_{id_name}.txt'
