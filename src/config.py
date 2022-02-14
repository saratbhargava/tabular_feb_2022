from hyperopt import hp
from hyperopt.pyll.base import scope

DEVICE = "cpu" # "cpu", "cuda"

TRAINING_FILE = "../input/train.csv"

TESTING_FILE = "../input/test.csv"

MODELS = "../models/"

SUBMIT = "../submit/"

N_SPLITS = 5

VALIDATION_TYPE = "StratifiedKfold" # "StratifiedKfold", "Kfold"

TARGET_LABEL = "target"


# Hyper parameter search with hyperopt
hyper_params = {
    "decision_tree": {
        "criterion": hp.choice("criteria", ["gini", "entropy"]),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    },
    "rf": {
        "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500, 700, 1000, 1200, 1300, 1500]),
        "criterion": hp.choice("criteria", ["gini", "entropy"]),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    },
    "svm": {
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        "loss": hp.choice("loss", ["hinge"]),
        "C": hp.loguniform("C", [-2, 8]),
    },
}
