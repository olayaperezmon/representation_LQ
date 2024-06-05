import itertools
from abc import ABC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from collections import Counter
import quapy as qp

class Phi(ABC):
    def fit(self, X, y):
        ...
    def transform(self, X):
        ...


class Phi_mean(Phi):
    def fit(self, X, y):
        return X
    
    def transform(self, X):
        return X.mean(axis=0)


class PhiPACC(Phi):
    def fit(self, X, y):
        self.phi = LogisticRegression()
        self.phi.fit(X, y)
        return self
    
    def transform(self, X):
        return self.phi.predict_proba(X).mean(axis=0)


class PhiACC(Phi):
    def fit(self, X, y):
        self.n_classes = np.unique(y)
        self.phi = LogisticRegression()
        self.phi.fit(X, y)
        return self
    
    def transform(self, X):
        pred = self.phi.predict(X)
        vect = np.zeros(len(self.n_classes))
        for i in self.n_classes:
            vect[i] = len(np.where(pred == i)[0])
        return vect/len(X)


class PhiHDy(Phi):
    def __init__ (self, n_bins):
        self.n_bins = n_bins 

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.phi = LogisticRegression()
        self.phi.fit(X, y)
        return self
    
    def transform(self, X):
        pred = self.phi.predict_proba(X)
        hist = np.zeros((self.n_classes, self.n_bins+1))
        for p in pred:
            bins = np.linspace(0, 1, self.n_bins+1)
            for i in range(self.n_classes):
                bin_index = np.digitize(p[i], bins) - 1
                hist[i, bin_index] = hist[i, bin_index] + 1
        res = hist/pred.shape[0]
        return res.flatten()


class Phi_most_voted(Phi):

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        def classifiers():
            yield LogisticRegression(max_iter=20000, solver='lbfgs'), "LogisticRegression", {'C': np.logspace(-3, 3, 7),'class_weight': [None, 'balanced']}
            yield SVC(max_iter=10000), "SVC", {'C': [0.1, 1, 10],'kernel': ['linear', 'rbf']}
            #yield SVM(max_iter = 10000), "SVM", {'C': [0.1, 1, 10],'kernel': ['linear', 'rbf']}
            yield RandomForestClassifier(), "RandomForest", {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'class_weight': [None, 'balanced']}

        self.grids = {}
        for classifier, name, param_grid in classifiers():
            grid = GridSearchCV(
                classifier,
                param_grid,
                cv=5,
                scoring=make_scorer(f1_score, average='macro'),
                n_jobs=-1
            ).fit(X, y)
            self.grids[name] = grid
        return self
        
    def transform(self, X):
        combined_predictions = []
        for name in self.grids:
            h = self.grids[name].predict(X)
            combined_predictions.append(h)
        pred_per_ex = list(zip(*combined_predictions))
        final_preds = []
        for ex in pred_per_ex:
            most_voted = Counter(ex).most_common(1)[0][0]
            final_preds.append(most_voted)
        vect = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            vect[i] = len(np.where(np.array(final_preds) == i)[0])
        return vect/len(X)


class Phi_most_prob(Phi):
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        def classifiers():
            yield LogisticRegression(max_iter=20000, solver='lbfgs'), "LogisticRegression", {'C': np.logspace(-3, 3, 7),'class_weight': [None, 'balanced']}
            yield SVC(probability=True, max_iter=10000), "SVC", {'C': [0.1, 1, 10],'kernel': ['linear', 'rbf']}
            yield RandomForestClassifier(), "RandomForest", {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'class_weight': [None, 'balanced']}

        self.grids = {}
        for classifier, name, param_grid in classifiers():
            grid = GridSearchCV(
                classifier,
                param_grid,
                cv=5,
                scoring=make_scorer(f1_score, average='macro'),
                n_jobs=-1
            ).fit(X, y)
            self.grids[name] = grid
        return self
        
    def transform(self, X):
        combined_probas = []
        for name in self.grids:
            h = self.grids[name].predict_proba(X)
            combined_probas.append(h)
        sum_probas = np.sum(combined_probas, axis=0)
        final_preds = np.argmax(sum_probas, axis=1)
    
        vect = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            vect[i] = len(np.where(final_preds == i)[0])
        return vect / len(X)


class Phi_Z_score(Phi):
    def __init__(self):
        pass

    def fit(self, X, y):
        lr = LogisticRegression()
        lr.fit(X, y)
        coefs = lr.coef_
        coefs[coefs < 0] = 0
        self.coefs = np.max(coefs, axis=0)
        return self
        
    def transform(self, X):
        Z = X * self.coefs
        ZTxZ = np.dot(Z.T, Z)
        upper_triangle = np.triu(ZTxZ)
        res = upper_triangle.flatten()
        return res


class Phi_classifiers_combination(Phi):

    def __init__(self, classifiers, acc_weight=False):
        self.classifiers = classifiers
        self.acc_weight = acc_weight

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))

        if self.acc_weight:
            self.accs = [np.mean(cross_val_score(c, X, y, n_jobs=-1)) for c in self.classifiers]
        else:
            self.accs = [1] * len(self.classifiers)
        self.accs = np.asarray(self.accs)
        self.accs /= self.accs.sum()

        for c in self.classifiers:
            c.fit(X, y)
        return self
        
    def transform(self, X):
        combined_probas = []
        for c, w in zip(self.classifiers, self.accs):
            h = (c.predict_proba(X)*w)
            combined_probas.append(h.mean(axis=0))
        combined_probas = np.concatenate(combined_probas)
        return combined_probas


class Phi_classifiers_grid(Phi):
    def __init__(self, classifier_cls, grid_hyperparams: dict, acc_weight=False):
        self.classsifier_cls = classifier_cls
        self.hyperparams = self.expand_grid(grid_hyperparams)
        self.acc_weight = acc_weight

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        classifiers = []
        for hyper in self.hyperparams:
            classifier = self.classsifier_cls(**hyper)
            classifiers.append(classifier)
        self.phi = Phi_classifiers_combination(classifiers, acc_weight=self.acc_weight)
        self.phi.fit(X, y)
        return self

    def expand_grid(self, param_grid: dict):
        params_keys = list(param_grid.keys())
        params_values = list(param_grid.values())
        configs = [{k: combs[i] for i, k in enumerate(params_keys)} for combs in itertools.product(*params_values)]
        return configs

    def transform(self, X):
        return self.phi.transform(X)

