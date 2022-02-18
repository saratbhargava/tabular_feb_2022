#/bin/bash

#########  PARAMTERS ######################

# select fold index
FOLD=2

# select ml model type
MODEL=xgb

# number of trails
NUM_TRAILS=1

# Submit message
SUBMIT_MESSAGE="XGB first submission using complete pipeline"

#########  TRAIN AND SUBMIT MODEL ###########

# Best Model
MODEL_FILENAME=${MODEL}_fold${FOLD}_`date +%m_%d_%Y_%H_%M_%S`.bin

SUBMIT_FILENAME=${MODEL_FILENAME:0:-4}.csv

echo Training ${MODEL} model started at `date`
SECONDS=0

# train the model
python train.py --fold ${FOLD} --model ${MODEL} --num_trails ${NUM_TRAILS} --model_filename ${MODEL_FILENAME}

train_duration=$SECONDS
echo Training ${MODEL} model complete at `date`
echo Training duration: ${train_duration} seconds

# test the model
echo Inference ${MODEL} model started at `date`
SECONDS=0

python inference.py --model_filename ${MODEL_FILENAME} --submit_filename ${SUBMIT_FILENAME}

inference_duration=$SECONDS
echo Inference ${MODEL} model complete at `date`
echo Inference duration: ${train_duration} seconds

# Submit the output csv to kaggle
kaggle competitions submit -c tabular-playground-series-feb-2022 -f ../submit/${SUBMIT_FILENAME} -m ${SUBMIT_MESSAGE}

echo Kaggle submission complete

##############################################
