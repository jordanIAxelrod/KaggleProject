"""
In this file I will select three models to train and cross validate.
Selection will be based on the best accuracy and the AUC.
I will be considering SVM, random forest, and gradientboost
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time


# given or using a predefined list of classifiers do a grid search cv and
# report the best accuracy and the parameters
class Learner:
    def __init__(self, X: pd.DataFrame, classifiers: list = None, params: dict = None) -> None:
        self.X = X
        # Assign classifiers if none are given
        if classifiers is None:
            self.classifiers = [
                GradientBoostingClassifier,
                SVC,
                RandomForestClassifier
            ]
        else:
            self.classifiers = classifiers
        # assign parameters for classifiers if none are given
        if params is None:
            self.params = {
                'GradientBoostingClassifier': {
                    'max_depth': [3, 10, 15, 20],
                    'loss': ['deviance', 'exponential'],
                    'n_estimators': [300, 400, 500, 600]
                },
                'SVC': {
                    'C': [1, 10],
                    'kernel': ['rbf', 'poly'],
                    'coef0': [0, 1, 2]
                },
                'RandomForestClassifier': {
                    'max_depth': [1, 2, 3, 10],
                    'n_estimators': [100, 200, 300, 400, 500]
                }
            }
        else:
            self.params = params

    def cross_validate(self, target: str) -> dict:
        best_params = {}
        # Split into test train
        X_train, X_test, y_train, y_test = train_test_split(
            self.X.drop(target, axis=1),
            self.X[target],
            test_size=.1,
            random_state=0
        )
        # iterate over classifiers
        for algo in self.classifiers:
            print(algo.__name__)
            # Record the start time
            start = time.time()
            # Run the grid search
            clf = GridSearchCV(
                algo(),
                self.params[algo.__name__],
                n_jobs=6
            )

            clf.fit(X_train, y_train)
            # Get the best parameters
            best_params[algo.__name__] = [
                clf.best_params_,
                clf.best_score_
            ]
            # Get the best estimator
            clf_best = clf.best_estimator_

            print(clf_best.score(X_test, y_test))
            print("Wall time:", time.time() - start, '/n')
        return best_params

