import os
print(os.system('hostname'))

import json
import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd

from config import dataConfig as dc
import controller as ctrl
from simulator import Simulator
from utils import init_seed, load_conf, SHORT_TO_FULL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate the online ranking system"
        "under differenct control policies"
    )

    parser.add_argument("conf", type=str, help="config file")
    parser.add_argument("dataset", type=str, help="dataset")
    parser.add_argument(
        "ctrl",
        type=str,
        choices=[
            "pc",
            "bpc",
            "bpcphi",
            "bpcphit",
            "ca",
            "cahinge",
            "oracle",
            "cp",
            "cphinge",
            "cpb",
            "cpbhinge",
            #"lpc",
            #"lpnc",
            #"smpc",
            #"smpca",
            #"olp",
            #"base",
            #"smpcc",
            #"onlinebpc",
            #"smpct_is",
            #"smpct_is_error",
            #"smpct_is_exact_hinge",
            #"smpct_is_approx_hinge",
            #"smpct_exact_c",
            #"onlinebpcnoerror",
            #"onlinebpcnoerrorutility",
            #"onlinebpcdist",
            #"oracle",
            #"bpc_relax",
            #"bpcphi",
            #"bpconline",
            #"onlinebpcnoerrorhinge",
            #"onlinebpcnophierrorhinge",
            #"bpcphit",
            #"smpct_is_approx_hinge_inside",
            #"smpct_is_error_hinge_inside",
            #"onlinebpcprojectnoerror",
            #"smpct_is_error_org"
        ],
        help="controller",
    )
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--lmbda", type=float, help="lambda used in P-Controller")
    parser.add_argument("--gamma", type=float, help="gamma used in BP-Controller")
    parser.add_argument("--horizon", type=int, help="horizon of the MPC")
    parser.add_argument(
        "--relearn", type=int, help="number of iterations to relearn in SMPC"
    )
    parser.add_argument(
        "--train_size", type=int, help="number of samples to train in SMPC"
    )
    parser.add_argument("--output_dir", type=str, help="Output of mterics")
    parser.add_argument("--c", nargs="+", help="C in weighted objective", type=float)
    parser.add_argument("--b", type=int, help="B offline in SMPC with bootstrap")
    parser.add_argument("--bo", type=int, help="B online in SMPC with bootstrap")
    parser.add_argument(
        "--relevance_type",
        type=str,
        choices=["offline_relevance", "online_relevance", 'sequence_relevance'],
        default=None,
    )
    parser.add_argument("--bpc_lr", type=float, help="OLP learning rate")
    parser.add_argument("--bpc_init", choices=["one", "zero"], help="OLP init value")
    parser.add_argument("--metrics_file_name", type=str, default=None)
    parser.add_argument("--targets", nargs="+", type=float)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--save_states", action="store_true")
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--hinge_min", type=float)
    parser.add_argument("--shuffle_bootstraps", type=str)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--beta", type=float)

    args = parser.parse_args()

    dc.init_config(args.dataset)
    N, MAX_UTIL = dc.N, dc.MAX_UTIL

    seed = args.seed if args.seed is not None else 0
    init_seed(seed)

    T, R, M, delta, config, m = load_conf(
        Path(args.conf), args.targets, args.dev, args.relevance_type
    )

    if args.lmbda is not None:
        config["lambda"] = args.lmbda

    if args.gamma is not None:
        config["gamma"] = args.gamma

    if args.horizon is not None:
        config["H"] = args.horizon

    if args.relearn is not None:
        config["relearn_iterations"] = args.relearn

    if args.train_size is not None:
        config["train_size"] = args.train_size

    if args.c is not None:
        config["C"] = args.c

    if args.b is not None:
        config["B"] = args.b

    if args.bo is not None:
        config["B_online"] = args.bo

    if args.bpc_lr is not None:
        config["bpc_lr"] = args.bpc_lr

    if args.bpc_init is not None:
        config["bpc_init"] = args.bpc_init

    if args.relevance_type is not None:
        config["relevance_type"] = args.relevance_type

    if args.metrics_file_name is not None:
        config["metrics_file_name"] = args.metrics_file_name
    else:
        config["metrics_file_name"] = None

    if args.momentum is not None:
        config['momentum'] = args.momentum

    if args.hinge_min is not None:
        config['hinge_min'] = args.hinge_min

    if args.shuffle_bootstraps is not None:
        config['shuffle_bootstraps'] = args.shuffle_bootstraps

    if args.beta is not None:
        config['beta'] = args.beta

    if args.eps is not None:
        config['eps'] = args.eps

    config["controller"] = SHORT_TO_FULL[args.ctrl]

    logger.level("DEBUG")

    logger.info(f"T={T}, N={N}, delta={delta}, config={config}")

    output_dir = f"{os.getcwd()}/results/{args.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ctrl_class = getattr(ctrl, config["controller"])
    controller = ctrl_class(T, N, M, delta, config)
    simulator = Simulator(
        controller, m, N, MAX_UTIL, delta, args.c, test=True, tqdm_disable=False
    )
    state, utility, obs = simulator.simulate(R)

    intermediate_metrics = simulator.intermediate_metrics
    df = pd.DataFrame(intermediate_metrics)
    if config["metrics_file_name"] is not None:
        df.to_pickle(f'{output_dir}/{config["metrics_file_name"]}.pkl')

    metrics = simulator.get_metrics(delta)

    columns = list(metrics.keys())
    values = [str(metrics[c]) for c in columns]
    header = ",".join(columns)
    print(header)
    row = ",".join(values)
    print(row)


    if args.output_dir is not None:
        output = dict(zip(columns, values))
        output_dir = Path(args.output_dir) / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.ctrl}"
        if "lambda" in config:
            run_name += f"_lambda_{config['lambda']}"
        if "gamma" in config:
            run_name += f"_gamma_{config['gamma']}"
        if "H" in config:
            run_name += f"_h_{config['H']}"
        if "relearn_iterations" in config:
            run_name += f"_relearn_{config['relearn_iterations']}"
        if "train_size" in config:
            run_name += f"_ts_{config['train_size']}"
        if "C" in config:
            run_name += f"_c_{'_'.join([str(x) for x in config['C']])}".replace(
                ".", "_"
            )
        if args.targets:
            run_name += f"_target_{args.targets[0]}"
        if args.ctrl == "smpca":
            if "B" in config:
                run_name += f"_b_{config['B']}"
            if "B_online" in config:
                run_name += f"_bo_{config['B_online']}"
        if args.bpc_init is not None:
            run_name += f"_init_{args.bpc_init}"
        if args.bpc_lr is not None:
            run_name += f"_lr_{args.bpc_lr}"
        if args.seed is not None:
            run_name += f"_seed_{args.seed}"
        run_name = run_name.replace(".", "_")
        output_path = output_dir / f"{run_name}.json"
        json.dump(output, output_path.open("w"))

        if args.save_states:
            states = simulator.get_states()
            states_output_path = output_dir / f"{run_name}.npy"
            np.save(states_output_path, states)
