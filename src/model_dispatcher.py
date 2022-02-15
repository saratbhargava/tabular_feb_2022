import config

from sklearn import ensemble, tree, svm

from xgboost import XGBClassifier

# Declare the various models
models = {
    "decision_tree": tree.DecisionTreeClassifier,
    "rf": ensemble.RandomForestClassifier,
    "xgb": XGBClassifier,
    "svm": svm.LinearSVC,
}
