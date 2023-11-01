"""Constants"""
from enum import Enum

N_BYTES_IN_MB = 1000 * 1000
ROUNDING_VAL = 6

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
        "nflow": 3000
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
