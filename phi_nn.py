from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quapy as qp
from quapy.protocol import UPP
import numpy as np
from quapy.method.base import BaseQuantifier
from abc import ABC
from quapy.data.base import LabelledCollection
from quapy.util import pickled_resource
from table import Table
import random
import wandb
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
import json
import warnings
warnings.simplefilter("ignore")


use_classification_loss = False
use_error_signal = False
use_autoencoder = False
use_triplet_loss = False
use_attention = False
use_regressor = False

#use_classification_loss = True
use_autoencoder = True
#use_attention = True
use_triplet_loss = True
use_regressor = True

show_lr = False

# wandb config
use_wandb = False
wandb_execution_name = "phi_nn_nuevo_triplet+reg+autoenc"


class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, Phi, solver="fro"):
        self.Phi = Phi
        self.solver = solver

    def fit(self, data: LabelledCollection):
        self.M, *_ = gen_M(data, self.Phi)
        self.n_classes = data.n_classes
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
        self.module.add_module("linear_1", nn.Linear(X_shape, X_shape*2))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("last_linear", nn.Linear(X_shape*2, X_shape//2))

    def dimensions(self):
        return self.module.last_linear.out_features

    def forward(self, x):
        return self.module(x)

class PhiPequena_decoder(nn.Module):
    def __init__(self, X_shape, last_linear_shape):
        super(PhiPequena_decoder, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1_dec", nn.Linear(last_linear_shape, X_shape*2))
        self.module.add_module("relu_1_dec", nn.ReLU())
        self.module.add_module("linear_2_dec", nn.Linear(X_shape*2, X_shape))

    def forward(self, x):
        return self.module(x)

class PhiGrande(nn.Module):
    def __init__(self, phi_pequena, use_attention):
        super(PhiGrande, self).__init__()
        self.phi_pequena = phi_pequena
        self.attention = nn.Parameter(torch.rand(latent_dims), requires_grad=True)
        self.use_attention = use_attention

    def forward(self, X):
        X = self.phi_pequena(X)
        mean = torch.mean(X, dim=0)
        if use_attention: 
            att = F.softmax(self.attention)
            mean_att = torch.mul(mean, att)
            return F.normalize(mean_att, p=2, dim=0), X 
        else: 
            return F.normalize(mean, p=2, dim=0), X

class Regressor_Mq(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor_Mq, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(input_dim, input_dim//2))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("linear_2", nn.Linear(input_dim//2, output_dim))

    def forward(self, x):
        return F.softmax(self.module(x))

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
    
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_json(q, M_prev, M_prev_contr):
    data_dict = {
        'q': q.tolist(),
        'M_prev': M_prev.tolist(),  
        'M_prev_contr': M_prev_contr.tolist() 
    }

    with open("data_dist.json", 'a') as f:
        json.dump(data_dict, f)
        f.write('\n')
    
def triplet_loss(prev, examples_prev_contr=50, mode='max'):
    # TRIPLET LOSS: min(|q - M@prev| - |q - M@prev_constr|)
    vect_matrix = qp.functional.uniform_prevalence_sampling(n_classes=num_classes, size=examples_prev_contr)
    p_aux = prev.reshape((1, -1))
    dist = distance.cdist(vect_matrix, p_aux, 'minkowski', p=1)

    if mode == 'max':
        ind = np.argmax(dist)
    elif mode == 'median':
        dist_sorted = np.stack(sorted(dist))
        ind = np.where(dist == dist_sorted[examples_prev_contr//2])[0][0]
    else:
        raise ValueError(f'unknown {mode=}')

    prev_contr = torch.tensor(vect_matrix[ind], dtype=torch.float32)
    prev = torch.tensor(prev, dtype=torch.float32)

    positive_dist = F.pairwise_distance(q, M@prev, p=2)
    negative_dist = F.pairwise_distance(q, M@prev_contr, p=2)

    factor = 1 #positive_dist.detach().numpy()/(2*negative_dist.detach().numpy())
    loss_triplet = positive_dist - factor*negative_dist
    return loss_triplet, prev_contr

class LossStr:
    def __init__(self, history=50):
        self.history = history
        self.losses = defaultdict(lambda :[])

    def add(self, loss, name):
        loss_ = loss.detach().numpy()
        self.losses[name].append(loss_)
        self.losses[name] = self.losses[name][-self.history:] 

    def __repr__(self):
        loss_str = []
        for loss_name in self.losses.keys():
            loss_str.append(f'{loss_name}={np.mean(self.losses[loss_name]):.6f}')
        return '\t'.join(loss_str)

def show_evaluation(train, test, repeats=100, prefix=''):
    rep = RepresentationLearningQuantification(phi_grande.eval(), "fro")
    rep.fit(train)
    report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=repeats), error_metrics=["mrae", "mae"])
    RAE = np.mean(report["mrae"].values)
    AE = np.mean(report["mae"].values)
    print(f'\t{prefix} : {RAE=:.5f}\t{AE=:.5f}')
    return RAE, AE

torch.manual_seed(2032)
np.random.seed(2032)
random.seed(2032)

if __name__ == '__main__':
    qp.environ['SAMPLE_SIZE'] = 250
    datasets = ['poker_hand'] #['dry-bean'] #['dry-bean','academic-success','digits','letter'] #, 'T1A']

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
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=5200, max_train_instances=800000) 
            qp.data.preprocessing.standardize(data, inplace=True)
            train, test = data.train_test

        #print(f'{len(train)=} {len(test)=}')
        
        train, val = train.split_stratified(train_prop=0.8)
        num_classes = len(train.classes_)

        def gen_M(data, phi_grande):
            M = []
            phi_Xs = []
            ys = []
            Xs = []
            for i in data.classes_:
                sel=data.y==i
                Xi = torch.tensor(data.X[sel], dtype=torch.float32)
                Phi_Xi, phi_Xi = phi_grande(Xi)
                Xs.append(Xi)
                M.append(Phi_Xi)
                phi_Xs.append(phi_Xi)
                ys.append(torch.full(size=(np.sum(sel),), fill_value= i))
            M = torch.stack(M).T
            phi_Xs = torch.concat(phi_Xs)
            Xs = torch.concat(Xs)
            ys = torch.concat(ys)

            if use_error_signal:
                M = M+e

            return M, phi_Xs, Xs, ys


        n_feat = train.X.shape[1]
        phi_pequena = PhiPequena(n_feat)
        latent_dims = phi_pequena.dimensions()
        phi_grande = PhiGrande(phi_pequena, use_attention)
        phi_pequena_decoder = PhiPequena_decoder(n_feat, latent_dims)
        e = nn.Parameter(torch.normal(mean=0.0, std=0.5, size=(latent_dims, num_classes)), requires_grad=True)
        regressor = Regressor_Mq(input_dim=latent_dims*num_classes+latent_dims, output_dim=num_classes)

        # params configuration
        params = list(phi_pequena.parameters())
        if use_classification_loss:
            linear_classif = nn.Linear(latent_dims, num_classes)
            params += list(linear_classif.parameters())
        if use_autoencoder:
            params += list(phi_pequena_decoder.parameters())
        if use_error_signal:
            params += [e]
        if use_attention:
            params += list(phi_grande.parameters())
        if use_regressor:
            params += list(regressor.parameters())

        optimizer = optim.Adam(params, lr=0.001) #, weight_decay=0.0001)
        
        early_stopping = EarlyStopping(patience=30)
        
        if show_lr:
            logr = LogisticRegression(max_iter=1000)
            logr.fit(train.X, train.y)

        n_ages = 1000
        mse_loss = nn.MSELoss()
        # loss_dec = None
        loss_str = LossStr()
        for age in range(n_ages):
            LM, Lq = train.split_stratified(train_prop=0.5)
            
            n_epochs = 5
            sample_size = random.randint(100, 500)
            for epoch in range(n_epochs):
                upp_gen = UPP(Lq, sample_size=sample_size, random_state=None, return_type="sample_prev")
                for i, (sam, prev) in enumerate(upp_gen()):
                    phi_pequena.train()
                    phi_pequena_decoder.train()
                    phi_grande.train()
                    regressor.train()
                    M, phi_X, X, y = gen_M(LM, phi_grande)
                    sam = torch.tensor(sam, dtype=torch.float32)
                    q, Xq = phi_grande(sam)
                    
                    # prev = torch.tensor(prev, dtype=torch.float32)
                    # loss_quant =  torch.norm(M@prev - q)

                    loss = 0  

                    if use_autoencoder:
                        phi_X_hat = phi_pequena_decoder(phi_X)
                        loss_autoencoder = mse_loss(phi_X_hat, X)
                        loss_str.add(loss_autoencoder, 'autoencoder')
                        loss += loss_autoencoder

                    if use_classification_loss:      
                        loss_classif = regularization_classif(phi_X, y, linear_classif)
                        loss_str.add(loss_classif, 'classif')
                        loss += loss_classif

                    if use_error_signal:
                        loss_e = torch.norm(e)
                        loss_str.add(loss_e, 'error')
                        loss += loss_e

                    if use_triplet_loss:
                        triplet, p_c = triplet_loss(prev, examples_prev_contr=50, mode='max')
                        loss_str.add(triplet, 'triplet')
                        loss += triplet

                    if use_regressor:
                        M_flat = M.flatten()
                        M_q = torch.concat((M_flat, q))
                        p_hat = regressor(M_q)
                        ae = torch.mean(torch.abs(torch.tensor(prev, dtype=torch.float32)-p_hat))
                        loss_str.add(ae, 'regressor ae')
                        loss += ae

                    loss_str.add(loss, 'total')
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                print(f'Age {age + 1}/{n_ages} : Epoch {epoch + 1}/{n_epochs}, Loss: {loss_str}')
                #save_json(q.detach().numpy(), (M@prev).detach().numpy(), (M@prev_contr).detach().numpy())
                

            show_evaluation(LM, Lq, prefix='Training')
            val_loss, _ = show_evaluation(train, val, prefix='Validation')
            show_evaluation(train+val, test, prefix='Test')

            """encoder = phi_pequena.eval()
            decoder = phi_pequena_decoder.eval()
            tensor_val = torch.tensor(val.X, dtype=torch.float32) 
            z = encoder(tensor_val)
            phi_X_hat_val = decoder(z)
            loss_autoencoder_val = mse_loss(phi_X_hat_val, tensor_val)
            print(f'\tValidation : {loss_autoencoder_val=:.5f}')"""

            if show_lr: 
                decision_func = logr.decision_function(val.X)
                ce_loss = nn.CrossEntropyLoss()
                ce_loss = ce_loss(torch.tensor(decision_func), torch.tensor(val.y))

            if use_wandb:
                logs = {"train_loss": loss.item(), "val_loss": val_loss}
                if use_autoencoder:
                    logs["loss_decoder"] = loss_autoencoder
                if use_classification_loss:
                    logs["loss_classif"] = loss_classif
                if use_error_signal:
                    logs["loss_error_signal"] = loss_e
                if use_triplet_loss:
                    logs["triplet_loss"] = triplet
                if use_regressor:
                    logs["loss_regressor"] = ae

                wandb.log(logs, step=age)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break


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
    
    #Table.LatexPDF('./latex/tablesPhiNN.pdf', [table_ae, table_rae], dedicated_pages=False)