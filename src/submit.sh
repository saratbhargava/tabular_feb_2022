#/bin/bash

# Submit the output csv to kaggle
kaggle competitions submit -c tabular-playground-series-feb-2022 -f ../submit/rf_fold1_12_02_2022_19_50_43.csv -m "First submition using a pipeline"
