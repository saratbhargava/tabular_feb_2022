import argparse
import joblib
from pathlib import Path

import config
import pandas as pd
from sklearn import preprocessing

from explainerdashboard import ClassifierExplainer, ExplainerDashboard


def run(fold, model_filename):

    # read the data
    df = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")

    # set the index
    df = df.set_index("row_id")
    df.index.name = config.INDEX_NAME

    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    df_train = df_train.drop(["fold"], axis=1)
    df_valid = df_valid.drop(["fold"], axis=1)

    # Create train features and target labels
    y_train = df_train[config.TARGET_LABEL]
    X_train = df_train.drop(config.TARGET_LABEL, axis=1)

    y_valid = df_valid[config.TARGET_LABEL]
    X_valid = df_valid.drop(config.TARGET_LABEL, axis=1)

    y_train.name = config.TARGET_NAME
    y_valid.name = config.TARGET_NAME

    # Load the model
    model_filepath = Path(config.MODELS) / model_filename
    model_obj = joblib.load(model_filepath)

    # Apply labelencoder
    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    y_train = le.transform(y_train)
    y_valid = le.transform(y_valid)

    # create an explainer board
    explainer = ClassifierExplainer(
        model_obj, X_valid, y_valid,
        labels=le.classes_)

    db = ExplainerDashboard(explainer, title="Bacteria Explainer",
                            whatif=False, # you can switch off tabs with bools
                            shap_interaction=False,
                            decision_trees=False)
    db.run(port=config.DASHBOARD_PORT)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explain ML model")

    parser.add_argument("--fold", type=int, help='Fold to use for validation')
    parser.add_argument("--model_filename", type=str, help='Model filename')

    args = parser.parse_args()

    run(fold=args.fold, model_filename=args.model_filename)
