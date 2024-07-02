import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quapy as qp
from quapy.protocol import UPP
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from quapy.method.base import BaseQuantifier
from abc import ABC
from quapy.data.base import LabelledCollection
from quapy.util import pickled_resource
from table import Table
import random
import wandb
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from phi_functions import PhiPACC
# from phi_functions import PhiPequena as PPeq
from representationLearningQuantification import run_experiment as runexp



class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, Phi, solver="fro"):
        self.Phi = Phi
        self.solver = solver

    def fit(self, data: LabelledCollection):
        self.M, *_ = gen_M(data, self.Phi)
        self.n_classes = data.n_classes
        """self.M = [self.Phi.transform(data.X[data.y==i]) for i in data.classes_]
        self.M = np.vstack(self.M).T"""
        return self

    def quantify(self, X):
        q, _ = self.Phi(torch.tensor(X, dtype=torch.float32)) 
        #Frobenius Distance
        if self.solver == "fro":
            def loss(p):
                return np.linalg.norm(self.M.detach().numpy() @ p - q.detach().numpy())
        
        #Hellinger Distance
        elif self.solver == "HD":
            def loss(p):
                M = 1/self.n_classes*self.M
                qq = 1/self.n_classes*q
                hdist = np.sqrt(np.sum((np.sqrt(p@M.T) - np.sqrt(qq))**2))
                return hdist
            
        return qp.functional.optim_minimize(loss, n_classes=self.n_classes)

class PhiPequena(nn.Module):
    def __init__(self, X_shape):
        super(PhiPequena, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(X_shape, X_shape))
        self.module.add_module("relu_1", nn.ReLU())
        # self.module.add_module("sig_1", nn.Tanh())
        # self.module.add_module("dropout_1", nn.Dropout(0.1))
        self.module.add_module("linear_2", nn.Linear(X_shape, X_shape//2))
        #self.module.add_module("relu_2", nn.ReLU())
        """self.module.add_module("linear_3", nn.Linear(X_shape//2, 32))
        self.module.add_module("relu_3", nn.ReLU())
        self.module.add_module("dropout_3", nn.Dropout(0.5))
        self.module.add_module("linear_4", nn.Linear(32, 10))"""
        #self.module.add_module("linear", nn.Linear(X_shape, X_shape//2))

    def forward(self, x):
        return self.module(x)

class PhiPequena_decoder(nn.Module):
    def __init__(self, X_shape):
        super(PhiPequena_decoder, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1_dec", nn.Linear(X_shape//2, X_shape))
        self.module.add_module("relu_1_dec", nn.ReLU())
        # self.module.add_module("sig_1_dec", nn.Tanh())
        # self.module.add_module("dropout_1_dec", nn.Dropout(0.1))
        self.module.add_module("linear_2_dec", nn.Linear(X_shape, X_shape))
        """self.module.add_module("relu_2_dec", nn.ReLU())
        self.module.add_module("dropout_2_dec", nn.Dropout(0.5))
        self.module.add_module("linear_3_dec", nn.Linear(64, 128))
        self.module.add_module("relu_3_dec", nn.ReLU())
        self.module.add_module("dropout_3_dec", nn.Dropout(0.5))
        self.module.add_module("linear_4_dec", nn.Linear(128, X_shape))"""
        #self.module.add_module("linear_dec", nn.Linear(X_shape//2, X_shape))

    def forward(self, x):
        return self.module(x)


class PhiGrande(nn.Module):
    def __init__(self, phi_pequena):
        super(PhiGrande, self).__init__()
        self.phi_pequena = phi_pequena

    def forward(self, X):
        X = self.phi_pequena(X)
        mean = torch.mean(X, dim=0)
        return F.normalize(mean, p=2, dim=0), X

def cosine_distance(x1, eps=1e-8):
    w1 = x1.norm(p=2, dim=0, keepdim=True)
    return (x1.t() @ x1) / (w1.t() * w1).clamp(min=eps)

def regularization_classif(X, y, linear_clas):
    X = linear_clas(X)
    cr_en_loss = nn.CrossEntropyLoss()
    loss = cr_en_loss(X, y)
    return loss

def run_experiment(train, test, phi):
    rep = RepresentationLearningQuantification(phi, "fro")
    rep.fit(train)
    report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=200), error_metrics=["mrae", "mae"])
    return report

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
    
        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


torch.manual_seed(2032)
np.random.seed(2032)
random.seed(2032)

use_wandb = False
use_classification_loss = False
show_lr = False
wandb_execution_name = "phi_nn_encoder_decoder_new"

if __name__ == '__main__':
    qp.environ['SAMPLE_SIZE'] = 250
    datasets = ['digits'] #,'academic-success','digits','letter'] #, 'T1A'] #'dry-bean',

    table_ae = Table(name='mae')
    table_rae = Table(name='mrae')
    
    for dataset_name in datasets:
        print(dataset_name)
        if use_wandb:
            wandb.login()
        if use_wandb:
            wandb.init(
                project="pruebas_phi_nn",
                name=dataset_name+'_'+wandb_execution_name,
                save_code=True,
            )

        if dataset_name == "T1A":
            train, _, test_gen = qp.datasets.fetch_lequa2022(dataset_name)
        else: 
            # train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset_name).train_test
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
            qp.data.preprocessing.standardize(data, inplace=True)
            train, test = data.train_test

        print(f'{len(train)=} {len(test)=}')
        
        train, val = train.split_stratified(train_prop=0.6)
        num_classes = len(train.classes_)


        def gen_M(data, phi_grande):
            M = []
            phi_X = []
            y = []
            X = []
            for i in data.classes_:
                X_i = torch.tensor(data.X[data.y==i], dtype=torch.float32)
                # phi_peqs = phi_pequena(X_i)
                # phi_gr = torch.mean(phi_peqs, dim=0)
                phi_gr, phi_peqs = phi_grande(X_i)
                X.append(X_i)
                M.append(phi_gr)
                phi_X.append(phi_peqs)
                y.append(torch.full(size=(np.sum(data.y == i),), fill_value= i))
            M = torch.stack(M).T 
            phi_X = torch.concat(phi_X)
            X = torch.concat(X)
            y = torch.concat(y)
            return M, phi_X, X, y

        n_feat = train.X.shape[1]
        #phi_pequena = PPeq(n_feat, num_classes)
        phi_pequena = PhiPequena(n_feat)
        phi_pequena_decoder = PhiPequena_decoder(n_feat)
        if use_classification_loss:
            linear_classif = nn.Linear(10, num_classes) 
            params = list(phi_pequena.parameters()) + list(linear_classif.parameters())
        else: 
            params = list(phi_pequena.parameters()) + list(phi_pequena_decoder.parameters())

        optimizer = optim.Adam(params, lr=0.00001) #, weight_decay=0.0001)
        
        early_stopping = EarlyStopping(patience=15)
        
        if show_lr:
            logr = LogisticRegression(max_iter=1000)
            logr.fit(train.X, train.y)

        n_ages = 500
        #weight_index = 0
        #tot_index = 40*100
        phi_grande = PhiGrande(phi_pequena)
        mse_loss = nn.MSELoss()
        for age in range(n_ages):
            LM, Lq = train.split_stratified(train_prop=0.5)
            #X_LM, y_LM = torch.tensor(LM.X, dtype=torch.float32), torch.tensor(LM.y, dtype=torch.int)
            #batch_size = 64 
            
            n_epochs = 50
            sample_size = random.randint(100, 500)
            for epoch in range(n_epochs):
                upp_gen = UPP(Lq, sample_size=sample_size, random_state=None, return_type="sample_prev")
                #weight_index = weight_index+1
                for i, (sam, prev) in enumerate(upp_gen()):
                    phi_pequena.train()
                    phi_pequena_decoder.train()
                    phi_grande.train()
                    M, phi_X, X, y = gen_M(LM, phi_grande)
                    # M, phi_X, X, y = gen_M(LM, None)
                    sam = torch.tensor(sam, dtype=torch.float32)
                    q, Xq = phi_grande(sam)
                    
                    # TRIPLET LOSS: min(|q - M@prev| - |q - M@prev_constr|)
                    vect_matrix = qp.functional.uniform_prevalence_sampling(n_classes = num_classes, size= 100)
                    p_aux = prev.reshape((1, -1))
                    dist = distance.cdist(vect_matrix, p_aux, 'minkowski', p=1)
                    ind = np.argmax(dist)
                    prev_contr = torch.tensor(vect_matrix[ind], dtype=torch.float32)
                    prev = torch.tensor(prev, dtype=torch.float32)
                    
                    positive_dist = F.pairwise_distance(q, M@prev, p=2)
                    negative_dist = F.pairwise_distance(q, M@prev_contr, p=2)
            
                    loss_triplet = positive_dist - negative_dist
                    # loss_contr = torch.norm(M@prev_contr - q)

                    # prev = torch.tensor(prev, dtype=torch.float32)
                    loss_quant =  torch.norm(M@prev - q)
                    

                    # LOSS DECODER : reconstruir X usando phi_pequeña_decoder
                    ## Esta X es la de LM aunque tb podria hacerlo con la de Lq, ns que es mejor
                    phi_x_hat = phi_pequena_decoder(phi_X)
                    loss_decoder = mse_loss(phi_x_hat, X)
                    

                    if use_classification_loss:      
                        loss_classif = regularization_classif(phi_X, y, linear_classif) 
                        # loss = loss_classif + loss_quant
                        #loss = (1-(weight_index/tot_index))*loss_classif+(weight_index/tot_index)*loss_rec
                    
                    loss =  loss_decoder + loss_triplet + 0*loss_quant
                    #loss = loss_reconstruction
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'Age {age+1}/{n_ages} : Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.5f}\t{loss_triplet=:.5f}\t{loss_decoder=:.5f}\t{loss_quant=:.5f}')


            LM_val, Lq_val = val.split_stratified(train_prop=0.5)
            rep = RepresentationLearningQuantification(phi_grande.eval(), "fro")
            rep.fit(LM_val)
            report = qp.evaluation.evaluation_report(rep, UPP(Lq_val, repeats=200), error_metrics=["mrae", "mae"])
            loss_val_rae = np.mean(report["mrae"].values)
            loss_val_ae = np.mean(report["mae"].values)
            print(f'VAL MRAE={loss_val_rae:.6f}\tMAE={loss_val_ae:.6f}')

            if show_lr: 
                decision_func = logr.decision_function(val.X)
                ce_loss = nn.CrossEntropyLoss()
                ce_loss = ce_loss(torch.tensor(decision_func), torch.tensor(val.y))

            # if use_wandb:
                #wandb.log({"train_loss": loss.item(), "val_loss": loss_val, "loss_classif": loss_classif, "loss_reconstruction": loss_quant, "LR_loss": ce_loss}, step=age)
                ## PARA TRIPLET LOSS:
                #wandb.log({"train_loss": loss.item(), "val_loss": loss_val, "loss_contr": loss_contr, "loss_qunat": loss_quant}, step=age)
                ## PARA LOSS ENCODER-DECODER:
                # wandb.log({"train_loss": loss.item(), "val_loss": loss_val, "loss_qunat": loss_quant, "loss_decoder": loss_decoder}, step=age)

            # early_stopping(loss_val)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        
        
        # LM_test, Lq_test = test.split_stratified(train_prop=0.5)
        # phi = phi_grande.eval()
        # path = f'results/{dataset_name}/phiNN__'+'.pkl'
        # result = pickled_resource(path, run_experiment, LM_test, Lq_test, phi)
        # aes = result["mae"].values
        # raes = result["mrae"].values

        # print(f'MAE={np.mean(aes):.6f}')
        # print(f'MRAE={np.mean(raes):.6f}')
        # print()
        # table_ae.add(benchmark=dataset_name, method='phiNN', v=aes)
        # table_rae.add(benchmark=dataset_name, method='phiNN', v=raes)
        # if use_wandb:
        #     wandb.finish()
    
    Table.LatexPDF('./latex/tablesPhiNN.pdf', [table_ae, table_rae], dedicated_pages=False)