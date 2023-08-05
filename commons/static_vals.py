"""Constants"""
from enum import Enum

N_BYTES_IN_MB = 1000 * 1000

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

# TODO: incomplete listing
# datasets for each modality
EXP_DATASETS = {
    DataModalities.TABULAR.value: ["adult"],
    DataModalities.SEQUENTIAL.value: ["nasdaq"],
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
        "dgan": 400
    }
}
