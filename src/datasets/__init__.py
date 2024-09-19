from enum import Enum


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validate"
    TEST = "test"
    PROD = "production"
