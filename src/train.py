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

from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

import wandb


def run(fold, model, tune, num_trails, model_filename):
    
    # read the data
    df = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")

    y = df[config.TARGET_LABEL]
    X = df.drop(["row_id", "fold"], axis=1)

    # Apply labelencoder
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    df_train = df_train.drop(["fold", "row_id"], axis=1)
    df_valid = df_valid.drop(["fold", "row_id"], axis=1)

    # Create train features and target labels
    y_train = df_train[config.TARGET_LABEL]
    X_train = df_train.drop(config.TARGET_LABEL, axis=1)

    y_valid = df_valid[config.TARGET_LABEL]
    X_valid = df_valid.drop(config.TARGET_LABEL, axis=1)

    feature_names = list(X_train.columns)

    # Apply labelencoder
    y_train = le.transform(y_train)
    y_valid = le.transform(y_valid)


    # hyper params optimization
    def objective(hyper_param_dict):
        model_class = model_dispatcher.models[model]
        model_obj = model_class(**hyper_param_dict)
        model_obj.fit(X_train, y_train)
        acc = model_obj.score(X_valid, y_valid)
        return {"loss": -acc, "status": STATUS_OK,
                "model": model_obj,
                "hyper_param_dict": hyper_param_dict}


    def k_fold_objective(hyper_param_dict):
        model_class = model_dispatcher.models[model]
        model_obj = model_class(**hyper_param_dict)
        cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True,
                                             random_state=config.RANDOM_STATE)
        cv_scores = model_selection.cross_val_score(
            estimator=model_obj, X=X, y=y,
            scoring='accuracy' cv=cv, n_jobs=-1)
        model_obj.fit(X, y)
        return {"loss": -np.mean(cv_scores), "status": STATUS_OK,
                "model": model_obj, "hyper_param_dict": hyper_param_dict}

    if fold < 0:
        objective_fn = k_fold_objective
    else:
        objective_fn = objective

    if tune:
        trials = Trials()
        best = fmin(
            fn = objective_fn,
            space = config.hyper_params[model], 
            algo = tpe.suggest,
            max_evals = num_trails,
            trials = trials,
        )

        for result in trials.results:
            run = wandb.init(project="Tabular_Feb2022", entity="sarat", reinit=True)
            current_model = result['model']
            wandb.config.update(result['hyper_param_dict'])
            wandb.log({'valid/accuracy': result['loss']})
            # visualize the model performance
            y_pred = current_model.predict(X_valid)
            y_pred_probas = current_model.predict_proba(X_valid)
            wandb.sklearn.plot_classifier(current_model,
                                          X_train, X_valid, y_train, y_valid,
                                          y_pred, y_pred_probas, np.unique(y_train),
                                          model_name=model, feature_names=feature_names)
            run.finish()

        # save the best model
        best_model = trials.results[np.argmin([result['loss'] for result in trials.results])]['model']
    else:
        best_model = objective_fn(config.fixed_hyper_params)['model']
    
    joblib.dump(
        best_model,
        Path(config.MODELS) / model_filename
    )
    
    return
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument("--fold", type=int, help='Fold to use for validation')
    parser.add_argument("--model", type=str, help='ML model',
                        choices=model_dispatcher.models.keys())
    parser.add_argument("--tune", action="store_true", help='Tune model parameters')
    parser.add_argument(
        "--num_trails", type=int,
        help='Number of trials for hyperparam tuning', default=3)
    parser.add_argument("--model_filename", type=str, help='Model filename')

    args = parser.parse_args()

    print(args.fold, args.model, args.num_trails, args.model_filename)
    
    run(fold=args.fold, model=args.model, tune=args.tune, num_trails=args.num_trails,
        model_filename=args.model_filename)
    
