# Ranking with exposure constraints

## Enviroment Setup
Python 3.7/8
- Create virtual Enviroment with virtualenv, conda or ...

- Install tensorflow that works on your machine

- Install required libraries

```shell
pip install -r requirements.txt
```
- Install pyxclib

```shell
# cd to any directory you prefer
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install 
```
## Usage
Copy `config.{xc,kuai}.yml.template` to `config.{xc,kuai}.yml` and update the original dataset path if dataset needs to be preprocessed.

- Data Preprocess (if necessary)

```shell
sh preprocess.sh
```

- Simulation Experiments

Create a experiment config file (see examples in `experiments/` folder) and replace the placeholder by the config path.

```shell
usage: simulate.py [-h] [--seed SEED] [--lmbda LMBDA] [--horizon HORIZON] [--relearn RELEARN] [--train_size TRAIN_SIZE] [--output_dir OUTPUT_DIR] [-c C] [-b B] [--bo BO] conf {xc,kuai} {pc,mpc,smpc,smpca}

Simulate the online ranking systemunder differenct control policies

positional arguments:
  conf                  config file
  {xc,kuai}             dataset
  {pc,mpc,smpc,smpca}   controller

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seed
  --lmbda LMBDA         lambda used in P-Controller
  --horizon HORIZON     horizon of the MPC
  --relearn RELEARN     number of iterations to relearn in SMPC
  --train_size TRAIN_SIZE
                        number of samples to train in SMPC
  --output_dir OUTPUT_DIR
                        Output of mterics
  -c C                  C in SMPC
  -b B                  B offline in SMPC with bootstrap
  --bo BO               B online in SMPC with bootstrap
```

### dataset
xc: Extreme Classification Dataset
kuai: KuaiRec

Example:
```shell
python -m simulate experiments/multi_group.yml xc smpc -c 0.3 --output_dir outputs/smpc_bo_ts -b 100 --bo 10 --train_size 600
```