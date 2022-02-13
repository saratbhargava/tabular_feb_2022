import config

if config.DEVICE == "cpu":
    from sklearn import tree
    from sklearn import ensemble
    from sklearn import svm
elif config.DEVICE == "cuda":
    from sklearn import tree  # cuml has no tree
    from cuml import ensemble
    from cuml import svm
else:
    raise ValueError(f"Invalid value for config.device: {config.device}")

# Declare the various models
models = {
    "decision_tree": tree.DecisionTreeClassifier,
    "rf": ensemble.RandomForestClassifier,
    "svm": svm.LinearSVC,
}


