import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# Our implementation of SEA algorithm

class SEA(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=None, metrics=accuracy_score, n_estimators=10):
        self.ensemble = []
        self.base_estimator = base_estimator
        self.metrics = metrics
        self.n_estimators = n_estimators

    def partial_fit(self, X, y, classes=None):
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.X_, self.y_ = X, y
        self.ensemble.append(clone(self.base_estimator).fit(self.X_, self.y_))

        # Remove worst estimator
        if len(self.ensemble) > self.n_estimators:
            del self.ensemble[
                np.argmin([self.metrics(y, clf.predict(X)) for clf in self.ensemble])
            ]
        return self

    def fit(self, X, y):
        self.partial_fit(X, y)
        return self

    def support_matrix(self, samples):
        ensemble = np.array([classificator.predict_proba(samples) for classificator in self.ensemble])
        return ensemble

    def predict_proba(self, samples):
        avg = np.mean(samples, axis=0)
        return avg

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        esm = self.support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
