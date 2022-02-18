import os

from hyperopt import hp
from hyperopt.pyll.base import scope


DEVICE = "cpu" # "cpu", "cuda"

TRAINING_FILE = "../input/train.csv"

TESTING_FILE = "../input/test.csv"

os.environ['WANDB_MODE'] = 'offline'

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
        "n_jobs": -1,
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    },
    "svm": {
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        # "loss": hp.choice("loss", ["hinge"]),
        "C": hp.loguniform("C", -2, 8),
    },
    "xgb":{
        "learning_rate": hp.choice("learning_rate", [0.01, 0.1, 0.3]),
        "n_estimators": hp.choice("n_estimators", [50, 100]),
        "max_depth": hp.choice("max_depth", [6, 10, 15]),
        "reg_lambda": hp.choice("reg_lambda", [0.01, 0.1, 1, 10]),
        "min_split_loss": hp.choice("min_split_loss", [0, 0.1, 0.2, 0.5]),
        "n_jobs": -1,
    }
}
