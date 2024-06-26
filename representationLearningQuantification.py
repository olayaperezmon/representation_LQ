from abc import ABC
import quapy as qp
from quapy.method.base import BaseQuantifier
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
import numpy as np
from quapy.method.aggregative import *
import quapy.functional as F
from quapy.protocol import UPP
from phi_functions import *
from quapy.util import pickled_resource
from table import Table


class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, Phi, solver="fro"):
        self.Phi = Phi
        self.solver = solver

    def fit(self, data: LabelledCollection):
        self.n_classes = data.n_classes
        self.M = [self.Phi.transform(data.X[data.y==i]) for i in data.classes_]
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
                M = 1/self.n_classes*self.M
                qq = 1/self.n_classes*q
                hdist = np.sqrt(np.sum((np.sqrt(p@M.T) - np.sqrt(qq))**2))
                return hdist
            
        return F.optim_minimize(loss, n_classes=self.n_classes)


def qp_quantifiers():
    yield ACC(LogisticRegression(max_iter=1000), val_split=0.4), "ACC"
    yield PACC(LogisticRegression(max_iter=1000), val_split=0.4), "PACC"
    yield DMy(LogisticRegression(max_iter=1000), val_split=0.4), "HDy"


classifiers = [LogisticRegression(max_iter=1000, C=1.0, class_weight=None), LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")]

lr_grid = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}


def representations():
    yield PhiACC(), "ACC"
    yield PhiPACC(), "PACC"
    yield PhiHDy(10), "HDy"
    #yield Phi_most_prob(), "Most probable"
    #yield Phi_most_voted(), "Most voted" 
    #yield Phi_Z_score(), "Z Score"
    #yield Phi_mean(), "Mean"
    #yield Phi_classifiers_combination(classifiers), "Combination of Classifiers"
    #yield Phi_classifiers_grid(LogisticRegression, lr_grid), "Grid of Classifiers"
    #yield Phi_classifiers_grid(LogisticRegression, lr_grid, acc_weight=True), "Grid of Classifiers Weighted"
    pass


# DEBUG = True
DEBUG = False
np.random.seed(2032)


def run_experiment(train, test, phi, name):
    phi.fit(train.X, train.y)
    if name == "HDy":
        rep = RepresentationLearningQuantification(phi, "HD")
    else:
        rep = RepresentationLearningQuantification(phi, "fro")
    rep.fit(train)
    report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=1000), error_metrics=["mrae", "mae"])
    return report

def run_experiment_quapy(train, test, quantifier, param_grid):
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
    report = qp.evaluation.evaluation_report(model, UPP(test, repeats=1000), error_metrics=["mrae", "mae"])
    return report

def run_experiment_quapy_no_ms(train, test, quantifier):
    model = quantifier.fit(train)
    report = qp.evaluation.evaluation_report(model, UPP(test, repeats=1000), error_metrics=["mrae", "mae"])
    return report


if __name__ == '__main__':
    
    qp.environ['SAMPLE_SIZE'] = 250
    uci_datasets = ['dry-bean','academic-success','digits','letter'] #'T1A

    table_ae = Table(name='mae')
    table_rae = Table(name='mrae')

    for dataset_name in uci_datasets:
        print(dataset_name)
        
        if DEBUG:
            train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).reduce().train_test
        else:
            train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).train_test
            train, val = train.split_stratified(train_prop=0.6)

        for phi, name in representations():
            path = f'results/{dataset_name}/{name}'+('__debug' if DEBUG else '')+'.pkl'
            result = pickled_resource(path, run_experiment, train, test, phi, name)
            
            aes = result["mae"].values
            raes = result["mrae"].values

            """print(f'Phi: {name}')
            print(f'MAE={np.mean(aes):.6f}')
            print(f'MRAE={np.mean(raes):.6f}')
            print()"""
            table_ae.add(benchmark=dataset_name, method=name+'(ours)', v=aes)
            table_rae.add(benchmark=dataset_name, method=name+'(ours)', v=raes)

        #print('-'*80)

        #Table.LatexPDF('./latex/tables.pdf', [table_ae, table_rae], dedicated_pages=False)"""
        
        
        ## QUAPY ##
        param_grid = {
                'classifier__C': np.logspace(-3, 3, 7),
                'classifier__class_weight': [None, 'balanced']
            }
            
        for quantifier, q_name in qp_quantifiers():
                # QUAPY WITHOUT MODEL SELECTION
                path = f'results/{dataset_name}/{q_name}'+'quapyNoMS'+('__debug' if DEBUG else '')+'.pkl'
                result_quapy_noms = pickled_resource(path, run_experiment_quapy_no_ms, train, test, quantifier)              
                aes_quapy_noms = result_quapy_noms["mae"].values
                raes_quapy_noms = result_quapy_noms["mrae"].values
                table_ae.add(benchmark=dataset_name, method=q_name+'(QuaPy without MS)', v=aes_quapy_noms)
                table_rae.add(benchmark=dataset_name, method=q_name+'(QuaPy without MS)', v=raes_quapy_noms)
                
                # QUAPY WITH MODEL SELECTION
                """path = f'results/{dataset_name}/{q_name}'+'quapyMS'+('__debug' if DEBUG else '')+'.pkl'
                result_quapy_ms = pickled_resource(path, run_experiment_quapy, train, test, quantifier, param_grid)             
                aes_quapy_ms = result_quapy_ms["mae"].values
                raes_quapy_ms = result_quapy_ms["mrae"].values
                table_ae.add(benchmark=dataset_name, method=q_name+'(QuaPy MS)', v=aes_quapy_ms)
                table_rae.add(benchmark=dataset_name, method=q_name+'(QuaPy MS)', v=raes_quapy_ms)"""


    Table.LatexPDF('./latex/tables.pdf', [table_ae, table_rae], dedicated_pages=False)