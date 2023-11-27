"""Constants"""
import torch
from enum import Enum

N_BYTES_IN_MB = 1000 * 1000
ROUNDING_VAL = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLTasks(Enum):
    """"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DataModalities(Enum):
    """"""
    TABULAR = "tabular"
    SEQUENTIAL = "sequential"
    TEXT = "text"


VALID_DATA_MODALITIES = [DataModalities.TABULAR.value,
                         DataModalities.SEQUENTIAL.value,
                         DataModalities.TEXT.value]

EXP_SYNTHESIZERS = {
    DataModalities.TABULAR.value: ["ctgan", "tvae", "gaussian_copula"],
    DataModalities.SEQUENTIAL.value: ["par", "dgan"],
    DataModalities.TEXT.value: ["gpt", "lstm"]
}

EXP_DATASETS = {
    DataModalities.TABULAR.value: ["adult", "census", "alarm", "child", "covtype", "credit",
                                   "expedia_hotel_logs", "insurance", "itrusion", "drugs"],
    DataModalities.SEQUENTIAL.value: ["nasdaq", "asu", "taxi"],
    DataModalities.TEXT.value: []
}

DEFAULT_EPOCH_VALUES = {
    "sdv": {
        "par": 128,
        "ctgan": 300,
        "tvae": 300,
        "gaussian_copula": 0
    },
    "gretel": {
        "dgan": 400,
        "actgan": 300
    },
    "synthcity": {
        "ddpm": 5000,
        "goggle": 300,
        "ctgan": 300,
        "rtvae": 300,
        "tvae": 300,
        "arf": 0,
        "nflow": 3000,
        "timegan": 1000
    },
    "ydata": {

    }
}

ML_CLASSIFICATION_TASK_DATASETS = [
    "adult", "census", "credit", "covtype", "loan", "intrusion"]
ML_REGRESSION_TASK_DATASETS = ["health_insurance"]
# drugs, child, car insurance

ML_TASKS_TARGET_CLASS = {
    "adult": "label",
    "census": "label",
    "credit": "label",
    "covtype": "label",
    "loan": "Personal Loan",
    "intrusion": "label",
    "health_insurance": "charges",
}

ML_CLASSIFICATION_MODELS = ["adaboost", "decision_tree", "logistic", "mlp"]
ML_REGRESSION_MODELS = ["linear", "mlp"]

SIMILARITY_CHECK_STATISTICS = ["mean", "median", "std"]


# ----------------
# reproducibility 
# ----------------
# stdlib
# import random

# import torch
# import numpy as np

# def enable_reproducible_results(random_state: int = 0) -> None:
#     np.random.seed(random_state)
#     try:
#         torch.manual_seed(random_state)
#     except BaseException:
#         pass
#     random.seed(random_state)
#     # TODO: Implement dgl seeding, like below:
#     # dgl.seed(random_state)

# def clear_cache() -> None:
#     try:
#         torch.cuda.empty_cache()
#     except BaseException:
#         pass

# distributions support 
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/core/distribution.py
