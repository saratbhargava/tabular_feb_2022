import config

from sklearn import ensemble, tree, svm

# Declare the various models
models = {
    "decision_tree": tree.DecisionTreeClassifier,
    "rf": ensemble.RandomForestClassifier,
    "svm": svm.LinearSVC,
}
