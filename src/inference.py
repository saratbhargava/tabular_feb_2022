import argparse
import joblib
from pathlib import Path

import pandas as pd

import config


def run(model_filename):

    # Load the data
    df = pd.read_csv(config.TESTING_FILE)
    row_ids = df['row_id']    
    df = df.drop("row_id", axis=1)

    X_test = df.values
    
    # Load the model
    model_filepath = Path(config.MODELS) / model_filename
    model_obj = joblib.load(model_filepath)

    # Predict the X_test
    y_test_pred = model_obj.predict(X_test)
    y_test_pred = pd.Series(y_test_pred, index=row_ids,
                            name=config.TARGET_LABEL)
    submit_filename = Path(config.SUBMIT) / model_filename[:-4]
    y_test_pred.to_csv(f"{submit_filename}.csv")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_filename", type=str)

    args = parser.parse_args()

    print(args.model_filename)
    
    run(model_filename=args.model_filename)
