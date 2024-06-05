from abc import ABC
import quapy as qp
from quapy.method.base import BaseQuantifier
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
import numpy as np
from quapy.method.aggregative import *
import quapy.functional as F
from quapy.protocol import APP, UPP
from phi_functions import *


class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, Phi, solver="fro"):
        self.Phi = Phi
        self.solver = solver

    def fit(self, data: LabelledCollection):
        self.n_classes = data.n_classes
        self.M = [self.Phi.transform(data.X[data.y==i])  for i in data.classes_]
        self.M = np.vstack(self.M).T
        return self

    def quantify(self, X):
        q = self.Phi.transform(X) 
        #Frobenius Distance
        if self.solver == "fro":
            def loss(p):
                return np.linalg.norm(self.M @ p - q)
        
        #Hellinger Distance
        elif self.solver == "HD":
            def loss(p):
                p = np.expand_dims(p, axis=0)
                mixture_distribution = (p @ self.M.reshape(self.n_classes,-1)).reshape(self.n_classes, -1)
                hdist = [np.sqrt(np.sum((np.sqrt(q[c]) - np.sqrt(mixture_distribution[c]))**2)) for c in range(self.n_classes)]
                return np.mean(hdist)
            
        return F.optim_minimize(loss, n_classes=self.n_classes)


def qp_quantifiers():
    yield ACC(LogisticRegression(max_iter=1000)), "ACC"
    yield PACC(LogisticRegression(max_iter=1000)), "PACC"
    yield DMy(LogisticRegression(max_iter=1000)), "HDy"

classifiers = [LogisticRegression(max_iter=1000, C=1.0, class_weight=None), LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")]

def representations():
    #yield PhiACC(), "ACC"
    yield PhiPACC(), "PACC"
    #yield PhiHDy(10), "HDy"
    #yield Phi_most_prob(), "Most probable"
    #yield Phi_most_voted(), "Most voted" 
    #yield Phi_Z_score(), "Z Score"
    #yield Phi_mean(), "Mean"
    yield Phi_classifiers_combination(classifiers), "Combination of Classifiers"

DEBUG = True
np.random.seed(2032)

#example
if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    uci_datasets = ['dry-bean','academic-success','digits'] #,'letter']
    
    for dataset_name in uci_datasets:
        
        if DEBUG:
            train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).reduce().train_test
        else:
            train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).train_test
        train, val = train.split_stratified(train_prop=0.6)
        
        for phi, name in representations():
            phi.fit(train.X, train.y)
            if name == "HDy":
                rep = RepresentationLearningQuantification(phi, "HD")    
            else:
                rep = RepresentationLearningQuantification(phi, "fro")
            rep.fit(train)
            p_hat = rep.quantify(test.X)
            print(f"{name} for {dataset_name}:")
            print("MRAE", qp.error.mrae(test.prevalence(), p_hat))
            print("MAE", qp.error.mae(test.prevalence(), p_hat))
        
        
        ## QUAPY ##
        """param_grid = {
        'classifier__C': np.logspace(-3, 3, 7),
        'classifier__class_weight': [None, 'balanced']
        }
        
        for quantifier, q_name in qp_quantifiers():
            optimizer = qp.model_selection.GridSearchQ( 
                        quantifier,
                        param_grid,
                        protocol=UPP(val, repeats=100),
                        error=qp.error.mrae,
                        refit=False,
                        verbose=True,
                        n_jobs=-1
                    ).fit(train)
            model = optimizer.best_model()
            report = qp.evaluation.evaluation_report(model, UPP(test, repeats=1000),error_metrics=["mrae", "mae"])
            print(report["mrae"].mean())"""