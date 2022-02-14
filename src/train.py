import argparse
from datetime import datetime
import joblib
from pathlib import Path

import config
import model_dispatcher

import hyperopt
from hyperopt import fmin, tpe, Trials, STATUS_OK

import numpy as np
import pandas as pd


if config.DEVICE == "cpu":    
    from sklearn import metrics
elif config.DEVICE == "cuda":
    from cuml import metrics
else:
    raise ValueError(f"Invalid value for config.device: {config.device}")


def run(fold, model, num_trails):
    
    # read the data
    df = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")
    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    df_train = df_train.drop(["fold", "row_id"], axis=1)
    df_valid = df_valid.drop(["fold", "row_id"], axis=1)

    # Create train features and target labels
    y_train = df_train[config.TARGET_LABEL]
    X_train = df_train.drop(config.TARGET_LABEL, axis=1)

    y_valid = df_valid[config.TARGET_LABEL]
    X_valid = df_valid.drop(config.TARGET_LABEL, axis=1)

    # hyper params optimization
    def objective(hyper_param_dict):
        model_class = model_dispatcher.models[model]
        model_obj = model_class(**hyper_param_dict, n_jobs=-1)
        model_obj.fit(X_train, y_train)
        acc = model_obj.score(X_valid, y_valid)
        return {"loss": -acc, "status": STATUS_OK, "model": model_obj}
    
    trials = Trials()
    best = fmin(
        fn = objective,
        space = config.hyper_params[model], 
        algo = tpe.suggest,
        max_evals = num_trails,
        trials = trials,
    )

    # save the best model
    best_model = trials.results[np.argmin([result['loss'] for result in trials.results])]['model']
    now = datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    joblib.dump(
        best_model,
        Path(config.MODELS) / f"{model}_fold{fold}_{now_str}.bin"
    )
    
    return
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument("--fold", type=int, help='Fold to use for validation')
    parser.add_argument("--model", type=str, help='ML model',
                        choices=model_dispatcher.models.keys())
    parser.add_argument(
        "--num_trails", type=int,
        help='Number of trials for hyperparam tuning', default=3)

    args = parser.parse_args()

    print(args.fold, args.model, args.num_trails)
    
    run(fold=args.fold, model=args.model, num_trails=args.num_trails)
    
