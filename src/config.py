from hyperopt import hp


TRAINING_FILE = "../input/train.csv"

MODEL_OUTPUT = "../models/"

N_SPLITS = 5

VALIDATION_TYPE = "StratifiedKfold" # "StratifiedKfold", "Kfold"

TARGET_LABEL = "target"


# Hyper parameter search with hyperopt
hyper_params = {
    "decision_tree": {
        "criteria": hp.choice("criteria", ["gini", "entropy"]),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    },
    "rf": {
        "n_estimators": hp.quniform("n_estimators", 10, 1_000, 1),
        "criteria": hp.choice("criteria", ["gini", "entropy"]),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    },
    "svm": {
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        "loss": hp.choice("loss", ["hinge"]),
        "C": hp.loguniform("C", [-2, 8]),
    },
}
