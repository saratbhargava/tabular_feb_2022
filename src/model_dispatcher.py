from sklearn import tree
from sklearn import ensemble
from sklearn import svm

# Declare the various models
models = {
    "decision_tree": tree.DecisionTreeClassifier(),
    "rf": ensemble.RandomForestClassifier(),
    "svm": svm.LinearSVC(),
}


