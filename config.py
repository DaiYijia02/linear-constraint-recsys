from pathlib import Path

import yaml


class DataConfig:
    DATA_DIR = None
    EXP_DIR = None
    K = None
    N = None
    MAX_UTIL = None
    DATASET = None
    IS_TEMPORAL = False

    def init_config(self, dataset):
        with open(Path(__file__).parent / f"config.{dataset}.yml", "r") as f:
            config = yaml.safe_load(f)
        self.DATASET = dataset
        self.DATA_DIR = Path(config["dataset"]["raw"]).expanduser()
        self.EXP_DIR = Path(config["dataset"]["exp"]).expanduser()
        self.EXP_DIR.mkdir(exist_ok=True)
        self.N = config["sample"].get("N", None)
        self.K = config["sample"].get("K", None)
        self.MAX_UTIL = config.get("max_util", None)
        self.IS_TEMPORAL = config.get("is_temporal", False)


dataConfig = DataConfig()
