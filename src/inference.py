import argparse
import joblib
from pathlib import Path

import pandas as pd

import config

from sklearn.preprocessing import LabelEncoder


def run(model_filename, submit_filename):

    # Load the testing data
    df = pd.read_csv(config.TESTING_FILE)
    row_ids = df['row_id']
    df = df.drop("row_id", axis=1)

    X_test = df.values

    # Load the training data
    df_train = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")
    y_train = df_train[config.TARGET_LABEL]
    
    # Load the model
    model_filepath = Path(config.MODELS) / model_filename
    model_obj = joblib.load(model_filepath)

    # Apply labelencoder
    le = LabelEncoder()
    le.fit(y_train)
    
    # Predict the X_test
    y_test_pred = model_obj.predict(X_test)
    y_test_pred = le.inverse_transform(y_test_pred)
    y_test_pred = pd.Series(y_test_pred, index=row_ids,
                            name=config.TARGET_LABEL)
    
    # Save the predictions
    submit_filepath = Path(config.SUBMIT) / submit_filename
    y_test_pred.to_csv(submit_filepath)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_filename", type=str)
    parser.add_argument("--submit_filename", type=str)

    args = parser.parse_args()

    run(model_filename=args.model_filename,
        submit_filename=args.submit_filename)
