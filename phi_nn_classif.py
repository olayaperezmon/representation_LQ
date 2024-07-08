from abc import ABC
import quapy as qp
from quapy.method.base import BaseQuantifier
from quapy.data.base import LabelledCollection
import numpy as np
from quapy.method.aggregative import *
import quapy.functional as F
from quapy.protocol import UPP
from phi_functions import *
from quapy.util import pickled_resource
from table import Table
import torch.optim as optim

class PhiPACC(Phi):
    def __init__(self):
        self.phipequena = None
    
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        n_classes = len(np.unique(y))
        self.phipequena = PhiPequena(X.shape[1], n_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.phipequena.parameters(), lr=0.00001) #, weight_decay=0.0001)
        
        self.phipequena.train()
        for epoch in range(100000):
            optimizer.zero_grad()
            output = self.phipequena(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print("Loss:", loss.detach().numpy())
        return self
    
    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.phipequena.eval()
        with torch.no_grad():
            probas = self.phipequena(X)
        return probas.mean(dim=0).numpy()

class PhiPequena(nn.Module):
    def __init__(self, X_shape, n_classes):
        super(PhiPequena, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(X_shape, X_shape*2))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("last_linear", nn.Linear(X_shape*2, 10))
        self.module.add_module("linear_last", nn.Linear(10, n_classes))
        

    def forward(self, x):
        x = self.module(x)
        return nn.functional.softmax(x,dim=1)


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


def representations():
    yield PhiPACC(), "PACC_withPhiPequena"
    pass

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

if __name__ == '__main__':
    
    qp.environ['SAMPLE_SIZE'] = 250
    uci_datasets = ['dry-bean'] #,'academic-success','digits','letter']

    table_ae = Table(name='mae')
    table_rae = Table(name='mrae')

    for dataset_name in uci_datasets:
        print(dataset_name)
        
        if DEBUG:
            train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).reduce().train_test
        else:
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
            qp.data.preprocessing.standardize(data, inplace=True)
            train, test = data.train_test

        for phi, name in representations():
            path = f'results/{dataset_name}/{name}'+('__debug' if DEBUG else '')+'.pkl'
            result = run_experiment(train, test, phi, name)

            aes = result["mae"].values
            raes = result["mrae"].values

            print(f'Phi: {name}')
            print(f'MAE={np.mean(aes):.6f}')
            print(f'MRAE={np.mean(raes):.6f}')
            print()

    #Table.LatexPDF('./latex/tables.pdf', [table_ae, table_rae], dedicated_pages=False)
