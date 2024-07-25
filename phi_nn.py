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
import argparse
import warnings
warnings.simplefilter("ignore")
from HardHistogram import HardHistogram 

TOCUDA=False
#TOCUDA=True

#optimization
USE_M_q = False
USE_TRIPLET_LOSS  = False
USE_REGRESSOR = False
USE_QUANT_LOSS = False

#variants
USE_CLASSIFICATION_POSTERIORS_LOSS = False
USE_CLASSIFICATION_NEXT_TO_LAST_LOSS = False
USE_AUTOENCODER  = False
USE_PHIGRANDE_COVMAT = False
USE_HISTNET = False

#hyperparam
USE_ATTENTION  = False


#USE_M_q = True

#USE_QUANT_LOSS = True
#USE_TRIPLET_LOSS  = True
#USE_REGRESSOR = True

#USE_CLASSIFICATION_POSTERIORS_LOSS = True
#USE_CLASSIFICATION_NEXT_TO_LAST_LOSS = True
#USE_AUTOENCODER  = True

#USE_PHIGRANDE_REGRESSOR = True
#USE_HISTNET = True
#USE_ATTENTION  = True

SHOW_LR = False

BINARY = False

HIDDEN = 100
NUM_TR_CLASS = 100

# wandb config
USE_WANDB = False
wandb_project_name = "NN_PHI"
#wandb_execution_name = "phi_nn_nuevo"+('_bin' if BINARY else '')+('+triplet' if USE_TRIPLET_LOSS else '')+('+cls' if USE_CLASSIFICATION_LOSS else '')+('+autoenc' if USE_AUTOENCODER else '')+('+reg' if USE_REGRESSOR else '')+('+att' if USE_ATTENTION else '')

"""if TOCUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(device)"""

def detach(tensor):
    if TOCUDA:
        return tensor.detach().cpu()
    else:
        return tensor.detach()

class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, Phi, solver="fro"):
        self.Phi = Phi
        self.solver = solver

    def fit(self, data: LabelledCollection):
        if USE_M_q:
            self.M, *_ = gen_M(data, self.Phi)
            self.n_classes = data.n_classes
            return self

    def quantify(self, X):

        if USE_M_q:
            q, _ = self.Phi(torch.tensor(X, dtype=torch.float32)) 

            if self.solver == "regressor": 
                regressor.eval()
                M_flat = self.M.flatten()
                M_q = torch.concat((M_flat, q))
                p_hat = regressor(M_q)
                return detach(p_hat).numpy()

            else:
                #Frobenius Distance
                if self.solver == "fro":
                    def loss(p):
                        return np.linalg.norm(detach(self.M).numpy() @ p - detach(q).numpy())
                
                #Hellinger Distance
                elif self.solver == "HD":
                    def loss(p):
                        M = 1/self.n_classes*self.M
                        qq = 1/self.n_classes*q
                        hdist = np.sqrt(np.sum((np.sqrt(p@M.T) - np.sqrt(qq))**2))
                        return hdist
                    
                return qp.functional.optim_minimize(loss, n_classes=self.n_classes)
        else:
            out, _ = self.Phi(torch.tensor(X, dtype=torch.float32)) 
            regressor.eval()    
            p_hat = regressor(out)
            return detach(p_hat).numpy()

class PhiPequena(nn.Module):
    def __init__(self, X_shape, phipeq_dropout, return_next_to_last):
        super(PhiPequena, self).__init__()
        self.return_next_to_last = return_next_to_last
        self.linear_1 = nn.Linear(X_shape, X_shape*2)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(phipeq_dropout)
        self.next_to_last_linear = nn.Linear(X_shape*2, X_shape//2)
        self.relu_2 = nn.ReLU()
        self.last_linear = nn.Linear(X_shape//2, num_classes)

    def dimensions(self):
        if self.return_next_to_last:
            return self.next_to_last_linear.out_features
        else:
            return self.last_linear.out_features 

    def forward(self, x): 
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.next_to_last_linear(x)

        if self.return_next_to_last:
            return x #return F.softmax(x) 
        else:
            x = self.relu_2(x)
            x = self.last_linear(x)
            return x

class PhiPequena_decoder(nn.Module):
    def __init__(self, X_shape, last_linear_shape):
        super(PhiPequena_decoder, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1_dec", nn.Linear(last_linear_shape, X_shape*2))
        self.module.add_module("relu_1_dec", nn.ReLU())
        self.module.add_module("dropout_1_dec", nn.Dropout(phipeq_droput))
        self.module.add_module("linear_2_dec", nn.Linear(X_shape*2, X_shape))

    def forward(self, x):
        return self.module(x)

"""class PhiGrande(ABC, nn.Module):
    def forward(self, X):
        ..."""

class PhiGrande_CovMat(nn.Module):
    def __init__(self, phi_pequena):
        super(PhiGrande_CovMat, self).__init__()
        self.phi_pequena = phi_pequena
        self.attention = nn.Parameter(torch.rand(latent_dims), requires_grad=True)

    def forward(self, X):
        X = self.phi_pequena(X)
        S = X.T@X
        S_n = S/X.shape[0]
        triu_ind = torch.triu_indices(S.shape[0], S.shape[0], 1)
        triu_S = S_n[triu_ind[0], triu_ind[1]]
        output = regressor_phigrande(triu_S)
        return output, X

class PhiGrande_Histograms(nn.Module):
    def __init__(self, phi_pequena):
        super(PhiGrande_Histograms, self).__init__()
        self.phi_pequena = phi_pequena
        self.attention = nn.Parameter(torch.rand(latent_dims), requires_grad=True)

    def forward(self, X):
        X = self.phi_pequena(X)
        hist = histnet(X)
        return hist, X

class PhiGrande_Mean(nn.Module):
    def __init__(self, phi_pequena, use_attention=False):
        super(PhiGrande_Mean, self).__init__()
        self.phi_pequena = phi_pequena
        self.attention = nn.Parameter(torch.rand(latent_dims), requires_grad=True)
        self.use_attention  = use_attention 

    def forward(self, X):
        X = self.phi_pequena(X)
        mean = torch.mean(X, dim=0)
        if self.use_attention : 
            att = F.softmax(self.attention)
            mean = torch.mul(mean, att)
            
        return mean, X #F.normalize(mean, p=2, dim=0), X

class HistNet_Layer(torch.nn.Module):
    def __init__(self, input_size, n_bins=32):
        super(HistNet_Layer, self).__init__()
        self.histogram = HardHistogram(
                        n_features=input_size,
                        num_bins=n_bins
                    )
        self.layers = torch.nn.Sequential()
        self.layers.add_module("sigmoid", torch.nn.Sigmoid())
        self.layers.add_module("histogram", self.histogram)
    
    def forward(self, x):
        output = self.layers(x)
        return output.squeeze(0)

class Regressor_PhiGrande(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor_PhiGrande, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear1", nn.Linear(input_dim, input_dim//2))
        self.module.add_module("relu1", nn.ReLU())
        self.module.add_module("droput1", nn.Dropout(reg_phigrande_dropout))
        self.module.add_module("linear2", nn.Linear(input_dim//2, output_dim))
        self.module.add_module("relu2", nn.ReLU())

    def forward(self, x):
        return self.module(x) #F.softmax(self.module(x)) #probar
    
class Regressor_Solver(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor_Solver, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(input_dim, input_dim//2))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("dropout_1", nn.Dropout(reg_dropout))
        self.module.add_module("linear_2", nn.Linear(input_dim//2, output_dim))

    def forward(self, x):
        return F.softmax(self.module(x))

def cosine_distance(x1, eps=1e-8):
    w1 = x1.norm(p=2, dim=0, keepdim=True)
    return (x1.t() @ x1) / (w1.t() * w1).clamp(min=eps)

def regularization_classif(X, y, linear_clas, next_to_last):
    if next_to_last:
        X = phi_pequena(X)
        X = linear_clas(X)

    else:
        X = phi_pequena(X)

    if BINARY: 
        y = y.view(-1, 1)
        bce_loss = nn.BCEWithLogitsLoss()
        loss = bce_loss(X, y)
    else:
        cr_en_loss = nn.CrossEntropyLoss()
        loss = cr_en_loss(X, y)
    return loss

def run_experiment(train, test, phi):
    if USE_REGRESSOR:
        rep = RepresentationLearningQuantification(phi, "regressor")
        
    else:
        rep = RepresentationLearningQuantification(phi, "fro")
    
    rep.fit(train)
    report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=200), error_metrics=["mrae", "mae"])
    return report

class EarlyStopping:
    def __init__(self, patience=20, lr_final = 0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.lr_final = lr_final

    def __call__(self, val_loss, test_report=None):
        lr = optimizer.param_groups[0]['lr']

        if lr <= self.lr_final:
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = val_loss
            if test_report is not None or not test_report.empty(): 
                #test_report.to_json(f"reports/{wandb_execution_name}.json")
                ty = "Mq" if USE_M_q else "Z"
                report = {
                    "mae_test": test_report["mae"].values.tolist(),
                    "mrae_test": test_report["mrae"].values.tolist(),
                    "val_loss": val_loss
                }
                #with open(f"rep/{ty}/{dataset_name}/{wandb_execution_name}.json", "w") as archivo:
                with open(f"prueba/{wandb_execution_name}.json", "w") as archivo:
                    archivo.write(json.dumps(report))
    
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0 
                #self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            if test_report is not None or not test_report.empty(): 
                ty = "Mq" if USE_M_q else "Z"
                report = {
                    "mae_test": test_report["mae"].values.tolist(),
                    "mrae_test": test_report["mrae"].values.tolist(),
                    "val_loss": val_loss
                }
                #with open(f"rep/{ty}/{dataset_name}/{wandb_execution_name}.json", "w") as archivo:
                with open(f"prueba/{wandb_execution_name}.json", "w") as archivo:
                    archivo.write(json.dumps(report))            

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
    def __init__(self, history=10):
        self.history = history
        self.losses = defaultdict(lambda :[])

    def add(self, loss, name):
        loss_ = detach(loss).numpy()
        self.losses[name].append(loss_)
        self.losses[name] = self.losses[name][-self.history:] 

    def __repr__(self):
        loss_str = []
        for loss_name in self.losses.keys():
            loss_str.append(f'{loss_name}={np.mean(self.losses[loss_name]):.6f}')
        return '\t'.join(loss_str)

def show_evaluation(train, test, repeats=100, prefix=''):
    if USE_M_q:
        if USE_REGRESSOR:
            rep = RepresentationLearningQuantification(phi_grande.eval(), "regressor")

        else:
            rep = RepresentationLearningQuantification(phi_grande.eval(), "fro")
        
        rep.fit(train)
    
    else:
        rep = RepresentationLearningQuantification(phi_grande.eval(), "regressor")
        rep.fit(LabelledCollection(instances = [], labels=[]))
        
    report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=repeats), error_metrics=["mrae", "mae"])
        
    RAE = np.mean(report["mrae"].values)
    AE = np.mean(report["mae"].values)
    print(f'\t{prefix} : {RAE=:.5f}\t{AE=:.5f}')
    return RAE, AE, report
    
torch.manual_seed(2032)
np.random.seed(2032)
random.seed(2032)

if __name__ == '__main__':
    qp.environ['SAMPLE_SIZE'] = 250

    parser = argparse.ArgumentParser(description='Training Phi NN')
    
    parser.add_argument("-mq", "--USE_M_q", type=bool, default=False)
    
    parser.add_argument("-reg", "--USE_REGRESSOR", type=bool, default=False)
    parser.add_argument("-trip", "--USE_TRIPLET_LOSS", type=bool, default=False)
    parser.add_argument("-quant", "--USE_QUANT_LOSS", type=bool, default=False)

    parser.add_argument("-cls_ntl", '--USE_CLASSIFICATION_NEXT_TO_LAST_LOSS', type=bool, default=False)
    parser.add_argument("-cls_post", '--USE_CLASSIFICATION_POSTERIORS_LOSS', type=bool, default=False)
    parser.add_argument("-reg_gr",'--USE_PHIGRANDE_REGRESSOR', type=bool, default=False)
    parser.add_argument("-hist",'--USE_HISTNET', type=bool, default=False)
    parser.add_argument("-aut", '--USE_AUTOENCODER', type=bool, default=False)

    parser.add_argument("-data", "--datasets", required=True)
    parser.add_argument("-d", "--device", help="Device cuda:0, cuda:1 or cpu", default="cpu")

    parser.add_argument("-dropout", "--dropout", type=bool, default=False)
    parser.add_argument("-att", "--USE_ATTENTION", type=bool, default=False)


    args = parser.parse_args()

    USE_M_q = args.USE_M_q
    USE_REGRESSOR = args.USE_REGRESSOR
    USE_TRIPLET_LOSS = args.USE_TRIPLET_LOSS
    USE_QUANT_LOSS = args.USE_QUANT_LOSS

    USE_CLASSIFICATION_POSTERIORS_LOSS = args.USE_CLASSIFICATION_POSTERIORS_LOSS
    USE_CLASSIFICATION_NEXT_TO_LAST_LOSS = args.USE_CLASSIFICATION_NEXT_TO_LAST_LOSS
    USE_PHIGRANDE_COVMAT = args.USE_PHIGRANDE_REGRESSOR
    USE_AUTOENCODER = args.USE_AUTOENCODER
    USE_HISTNET = args.USE_HISTNET
    USE_ATTENTION = args.USE_ATTENTION

    if args.dropout:
        phipeq_droput = 0.5
        reg_dropout = 0.5
        reg_phigrande_dropout = 0.5
    else:
        phipeq_droput = 0
        reg_dropout = 0
        reg_phigrande_dropout = 0

    device = torch.device(args.device)
    datasets = [args.datasets]

    print(args)

    if args.device != 'cpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(device)
        TOCUDA = True

    """if BINARY: 
        datasets = ['wine-q-white']
    else:     
        datasets = ['connect-4'] #['poker_hand'] #['dry-bean'] #['dry-bean','academic-success','digits','letter'] #, 'T1A']"""
    
    for dataset_name in datasets:
        print(dataset_name)
        wandb_execution_name = dataset_name+'_'+('Mq+' if USE_M_q else '')+('REG' if USE_REGRESSOR else '')+('QNT' if USE_QUANT_LOSS else '')+('TRI' if USE_TRIPLET_LOSS else '')+('+CP1' if USE_CLASSIFICATION_POSTERIORS_LOSS else '+CP0')+('+CN1' if USE_CLASSIFICATION_NEXT_TO_LAST_LOSS else '+CN0')+('+AE1' if USE_AUTOENCODER else '+AE0')+('+PGR1' if USE_PHIGRANDE_COVMAT else '+PGR0')+('+16bHIS1' if USE_HISTNET else '+HIS0')+('+ATT1' if USE_ATTENTION else '+ATT0')+('+DO1' if args.dropout else '+DO0')
        if USE_WANDB:
            wandb.login()
        if USE_WANDB:
            wandb.init(
                project=wandb_project_name,
                name=wandb_execution_name,
                save_code=True,
            )

        if dataset_name == "connect-4":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, max_train_instances=47290)  
        elif dataset_name == "chess":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=251)
        elif dataset_name == "dry-bean" or dataset_name == "hand_digits":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
        elif dataset_name == "shuttle":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=200)
        elif dataset_name == "poker_hand":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=4100, max_train_instances=800000) 

        qp.data.preprocessing.standardize(data, inplace=True)
        train, test = data.train_test
        
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

            return M, phi_Xs, Xs, ys

        n_feat = train.X.shape[1]
        phi_pequena = PhiPequena(n_feat, phipeq_droput, return_next_to_last=not USE_CLASSIFICATION_POSTERIORS_LOSS)
        latent_dims = phi_pequena.dimensions()
        #phi_pequena_decoder = PhiPequena_decoder(n_feat, latent_dims)
        
        #regressor = Regressor_Mq(input_dim=latent_dims*num_classes+latent_dims, output_dim=num_classes)
        #regressor_phigrande = Regressor_PhiGrande(latent_dims*(latent_dims-1)//2, latent_dims)
        #histnet = HistNet_Layer(input_size = latent_dims, n_bins = 8)

        #phi_grande_covmat = PhiGrande_CovMat(phi_pequena)
        #phi_grande_mean = PhiGrande_Mean(phi_pequena, USE_ATTENTION)
        #phi_grande_histograms = PhiGrande_Histograms(phi_pequena)

        # params configuration
        p = []
        p.append("phi_pequena")
        params = list(phi_pequena.parameters())
        
        if USE_CLASSIFICATION_NEXT_TO_LAST_LOSS:
            """if BINARY: 
                linear_classif = nn.Linear(latent_dims, num_classes-1)
            else: """
            linear_classif = nn.Linear(latent_dims, num_classes)
            
            params += list(linear_classif.parameters())
            p.append("linear_classif")
        if USE_AUTOENCODER :
            phi_pequena_decoder = PhiPequena_decoder(n_feat, latent_dims)
            params += list(phi_pequena_decoder.parameters())
            p.append("phi_pequena_decoder")
        if USE_ATTENTION :
            phi_grande = PhiGrande_Mean(phi_pequena, USE_ATTENTION)
            params += list(phi_grande.parameters())
            p.append("phi_grande_att")
        if USE_HISTNET:
            n_bins = 16
            histnet = HistNet_Layer(input_size = latent_dims, n_bins = n_bins)
            phi_grande = PhiGrande_Histograms(phi_pequena)
            params += list(histnet.parameters())
            params += list(phi_grande.parameters()) 
            p.append("phi_gr_hist")
            p.append("hist")
        if USE_REGRESSOR:
            if USE_HISTNET:
                regressor = Regressor_Solver(input_dim=(latent_dims*num_classes+latent_dims)*n_bins, output_dim=num_classes)
            else:
                regressor = Regressor_Solver(input_dim=latent_dims*num_classes+latent_dims, output_dim=num_classes)
            params += list(regressor.parameters())
            p.append("regressor")
        if USE_PHIGRANDE_COVMAT: 
            regressor_phigrande = Regressor_PhiGrande(latent_dims*(latent_dims-1)//2, latent_dims)
            phi_grande = PhiGrande_CovMat(phi_pequena)
            params += list(regressor_phigrande.parameters())
            params += list(phi_grande.parameters())
            p.append("regressor_phigrande")
            p.append("phi_grcovmat")
        if not USE_HISTNET and not USE_PHIGRANDE_COVMAT and not USE_ATTENTION:
            phi_grande = PhiGrande_Mean(phi_pequena, USE_ATTENTION)
            params += list(phi_grande.parameters())
            p.append("phi_gr_mean")
        if not USE_M_q:
            if USE_HISTNET:
                regressor = Regressor_Solver(input_dim=latent_dims*n_bins, output_dim=num_classes)
            else:
                regressor = Regressor_Solver(input_dim=latent_dims, output_dim=num_classes)
            params += list(regressor.parameters())
            p.append("reg")
        print("params", p)
        optimizer = optim.Adam(params, lr=0.001) #, weight_decay=0.0001)
        
        patience = 20
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5, cooldown=0, verbose=True)
        
        early_stopping = EarlyStopping(patience=patience, lr_final=0.0001)
        
        if SHOW_LR:
            logr = LogisticRegression(max_iter=1000)
            logr.fit(train.X, train.y)

        n_ages = 1000
        mse_loss = nn.MSELoss()
        loss_str = LossStr()
        n = 20000
        min_val_loss = np.inf
        for age in range(n_ages):

            if USE_M_q :
                LM, Lq = train.split_stratified(train_prop=0.5)
                sample_size = random.randint(100, 500)
                if len(LM)> n: 
                    LM = LM.sampling(n, *LM.prevalence())
                
                upp_gen = UPP(Lq, sample_size=sample_size, random_state=None, return_type="sample_prev")
                for i, (sam, prev) in enumerate(upp_gen()):
                    phi_pequena.train()
                    phi_grande.train()

                    if USE_AUTOENCODER:
                        phi_pequena_decoder.train()
                    if USE_REGRESSOR:
                        regressor.train()
                    if USE_PHIGRANDE_COVMAT:
                        regressor_phigrande.train()
                    if USE_HISTNET:
                        histnet.train()
                    
                    M, phi_X, X, y = gen_M(LM, phi_grande)

                    sam = torch.tensor(sam, dtype=torch.float32)
                    q, Xq = phi_grande(sam)
                        
                    loss = 0 

                    if USE_AUTOENCODER :
                        phi_X_hat = phi_pequena_decoder(phi_X)
                        loss_autoencoder = mse_loss(phi_X_hat, X)
                        loss_str.add(loss_autoencoder, 'autoencoder')
                        loss += loss_autoencoder

                    if USE_CLASSIFICATION_NEXT_TO_LAST_LOSS:    
                        loss_classif_ntl = regularization_classif(X, y, linear_classif, next_to_last=True)
                        loss_str.add(loss_classif_ntl, 'classif_ntl')
                        loss += loss_classif_ntl
                        
                    if USE_CLASSIFICATION_POSTERIORS_LOSS:    
                        loss_classif_post = regularization_classif(X, y, None, next_to_last=False)
                        loss_str.add(loss_classif_post, 'classif_post')
                        loss += loss_classif_post

                    if USE_TRIPLET_LOSS:
                        triplet, p_c = triplet_loss(prev, examples_prev_contr=50, mode='max')
                        loss_str.add(triplet, 'triplet')
                        loss += triplet

                    if USE_QUANT_LOSS:
                        prev = torch.tensor(prev, dtype=torch.float32)
                        quant_loss = torch.norm(M@prev-q)
                        loss_str.add(quant_loss, 'quant_loss')
                        loss += quant_loss

                    if USE_REGRESSOR:
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

            else: 
                sample_size = random.randint(100, 500)
                
                upp_gen = UPP(train, sample_size=sample_size, random_state=None, return_type="sample_prev")
                for i, (sam, prev) in enumerate(upp_gen()):
                    phi_pequena.train()
                    phi_grande.train()
                    regressor.train()

                    if USE_AUTOENCODER:
                        phi_pequena_decoder.train()
                    if USE_PHIGRANDE_COVMAT:
                        regressor_phigrande.train()
                    if USE_HISTNET:
                        histnet.train()

                    sam = torch.tensor(sam, dtype=torch.float32)
                    Phi_X, phi_X = phi_grande(sam)
            
                    loss = 0

                    prev = torch.tensor(prev, dtype=torch.float32)
                    p_hat = regressor(Phi_X)
                    ae = torch.mean(torch.abs(prev-p_hat))
                    loss_str.add(ae, 'regressor solver ae')
                    loss += ae 
                
                    if USE_AUTOENCODER :
                        phi_X_hat = phi_pequena_decoder(phi_X)
                        loss_autoencoder = mse_loss(phi_X_hat, sam)
                        loss_str.add(loss_autoencoder, 'autoencoder')
                        loss += loss_autoencoder

                    if USE_CLASSIFICATION_NEXT_TO_LAST_LOSS:    
                        loss_classif_ntl = regularization_classif(sam, prev, linear_classif, next_to_last=True)
                        loss_str.add(loss_classif_ntl, 'classif_ntl')
                        loss += loss_classif_ntl
                        
                    if USE_CLASSIFICATION_POSTERIORS_LOSS:    
                        loss_classif_post = regularization_classif(sam, prev, None, next_to_last=False)
                        loss_str.add(loss_classif_post, 'classif_post')
                        loss += loss_classif_post

                    loss_str.add(loss, 'total')
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f'Epoch {age + 1}/{n_ages}, Loss: {loss_str}')
        
            if USE_M_q:        
                show_evaluation(LM, Lq, prefix='Training')
            else:
                show_evaluation([], train, prefix='Training')
            
            val_rae_loss, val_ae_loss, _ = show_evaluation(train, val, prefix='Validation')
            if val_ae_loss < min_val_loss:
                min_val_loss = val_ae_loss
                _, _, test_report = show_evaluation(train+val, test, prefix='Test')

            if SHOW_LR: 
                decision_func = logr.decision_function(val.X)
                ce_loss = nn.CrossEntropyLoss()
                ce_loss = ce_loss(torch.tensor(decision_func), torch.tensor(val.y))

            if USE_WANDB:
                logs = {"train_loss": loss.item(), "val_loss": val_rae_loss, "val_AE_loss": val_ae_loss}
                if USE_AUTOENCODER :
                    logs["loss_decoder"] = loss_autoencoder
                if USE_CLASSIFICATION_NEXT_TO_LAST_LOSS:
                    logs["loss_classif_ntl"] = loss_classif_ntl
                if USE_CLASSIFICATION_POSTERIORS_LOSS:
                    logs["loss_classif_post"] = loss_classif_post
                if USE_TRIPLET_LOSS :
                    logs["triplet_loss"] = triplet
                if USE_QUANT_LOSS:
                    logs["quant_loss"] = quant_loss
                if USE_REGRESSOR:
                    logs["loss_regressor"] = ae

                wandb.log(logs, step=age)

            scheduler.step(val_ae_loss)
            early_stopping(val_ae_loss, test_report)
            if early_stopping.early_stop:
                print("Early stopping")
                break