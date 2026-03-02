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
import os
import warnings
warnings.simplefilter("ignore")
from HardHistogram import HardHistogram 
from GMLayer import GMLayer
from MeanLayer import MeanLayer

from quapy.data.datasets import fetch_lequa2022, LEQUA2022_SAMPLE_SIZE
from quapy.data.datasets import fetch_lequa2024, LEQUA2024_SAMPLE_SIZE
from quapy.protocol import AbstractProtocol
import torchvision
from transformers import DistilBertModel, DistilBertTokenizerFast
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import geotorch
import scipy

# wandb config
USE_WANDB = True
BINARY = ["T1", "T1A", "T2A"]
IMAGES = ["CIFAR10", "CIFAR100coarse"]

class RepresentationLearningQuantification(BaseQuantifier, ABC):
    def __init__(self, BagRepresentation, regressor, use_Mq, solver="fro"):
        self.BagRepresentation = BagRepresentation
        self.solver = solver
        self.use_Mq = use_Mq
        self.regressor = regressor


    def fit(self, data: LabelledCollection):
        if self.use_Mq:
            self.M, *_ = gen_M(data, self.BagRepresentation, chunk_size=512, train=False)
            self.n_classes = data.n_classes
        return self

    def predict(self, X):
        
        if self.use_Mq:
            dev = next(self.BagRepresentation.parameters()).device
            q, *_ = self.BagRepresentation(torch.tensor(X, dtype=torch.float32, device=dev))

            if self.solver == "regressor": 
                self.regressor.eval()
                M_flat = self.M.flatten()
                M_q = torch.cat((M_flat, q))
                p_hat = self.regressor(M_q, softmax=True)
                return detach(p_hat).numpy()

            else:
                #Frobenius Distance
                if self.solver == "fro":
                    M = self.M.detach().cpu().numpy()
                    q = q.detach().cpu().numpy()
                    def loss(p):
                        return np.linalg.norm(M @ p - q)
                
                #Hellinger Distance
                elif self.solver == "HD":
                    def loss(p):
                        M = 1/self.n_classes*self.M
                        qq = 1/self.n_classes*q
                        hdist = np.sqrt(np.sum((np.sqrt(p@M.T) - np.sqrt(qq))**2))
                        return hdist

                return qp.functional.optim_minimize(loss, n_classes=self.n_classes)  

        else:
            dev = next(self.BagRepresentation.parameters()).device
            out, *_ = self.BagRepresentation(torch.tensor(X, dtype=torch.float32, device=dev))
            self.regressor.eval()    
            p_hat = self.regressor(out, softmax=True)
            return detach(p_hat).numpy()

class InstanceRepresentation(nn.Module):
    """
    This class defines the representation of the individual isntances. 
    In general (phi_l, phi_f, phi_A), it return the next_to_last linar infromation 
    except when we use a traditional classifier (phi_h) that returns the posteriors. 
    """

    def __init__(self, X_shape, dropout, return_next_to_last, num_classes=None):
        super(InstanceRepresentation, self).__init__()
        self.return_next_to_last = return_next_to_last
        self.linear_1 = nn.Linear(X_shape, X_shape*2)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.next_to_last_linear = nn.Linear(X_shape*2, X_shape//2)
        if num_classes != None: 
            self.relu_2 = nn.ReLU()
            if num_classes == 2:
                self.last_linear = nn.Linear(X_shape//2, num_classes-1)
            else:
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
            return x, None 
        else:
            x = self.relu_2(x)
            x = self.last_linear(x)
            return F.softmax(x, dim=-1), x   
    
class InstanceRepresentationImages(nn.Module):
    def __init__(self, in_channels, latent_dim, return_next_to_last, num_classes=None, img_size=32):
        super(InstanceRepresentationImages, self).__init__()
        self.return_next_to_last = return_next_to_last
        self.img_size = img_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),          
            nn.ReLU(),
        )

        self.enc_h = img_size // 4   
        self.enc_w = img_size // 4
        self.flat_dim = 128 * self.enc_h * self.enc_w

        self.next_to_last_linear = nn.Linear(self.flat_dim, latent_dim)

        if num_classes != None: 
            self.relu_2 = nn.ReLU()
            if num_classes == 2:
                self.last_linear = nn.Linear(latent_dim, num_classes-1)
            else:
                self.last_linear = nn.Linear(latent_dim, num_classes)

    def dimensions(self):
        if self.return_next_to_last:
            return self.next_to_last_linear.out_features
        else:
            return self.last_linear.out_features 

    def forward(self, x): 
        x = self.features(x)
        x = x.view(x.size(0), -1)   
        x = self.next_to_last_linear(x)  

        if self.return_next_to_last:
            return x, None 
        else:
            x = self.relu_2(x)
            x = self.last_linear(x)
            return F.softmax(x, dim=-1), x   

class InstanceRepresentationImages_decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, img_size=32):
        super(InstanceRepresentationImages_decoder, self).__init__()
        self.img_size = img_size
        self.enc_h = img_size // 4   
        self.enc_w = img_size // 4
        self.flat_dim = 128 * self.enc_h * self.enc_w

        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.enc_h, self.enc_w)  
        x = self.decoder(x)                                 
        return x

class InstanceRepresentation_decoder(nn.Module):
    def __init__(self, X_shape, last_linear_shape, dropout):
        super(InstanceRepresentation_decoder, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1_dec", nn.Linear(last_linear_shape, X_shape*2))
        self.module.add_module("relu_1_dec", nn.ReLU())
        self.module.add_module("dropout_1_dec", nn.Dropout(dropout))
        self.module.add_module("linear_2_dec", nn.Linear(X_shape*2, X_shape))

    def forward(self, x):
        return self.module(x)

class phi_bert(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.linear = nn.Linear(768, output_size)

        for p in self.bert.parameters():
            p.requires_grad = False

    def dimensions(self):
        return self.linear.out_features

    def forward(self, x):
        out = self.bert(**x)
        features = out[0][:, 0, :]  
        return self.linear(features)


class BagRepresentation_CovMat(nn.Module):
    def __init__(self, instance_representation, regressor):
        super(BagRepresentation_CovMat, self).__init__()
        self.instance_representation = instance_representation
        self.regressor = regressor
    
    def forward(self, X):
        X, X_logits = self.instance_representation(X)
        mean = torch.mean(X, dim=0)
        cov_mat = torch.cov(X.T)
        triu_ind = torch.triu_indices(cov_mat.shape[0], cov_mat.shape[1], 0)
        triu_covmat = cov_mat[triu_ind[0], triu_ind[1]]
        triu_covmat = triu_covmat.flatten()

        covmat_mean = torch.cat((triu_covmat, mean))
        output = self.regressor(covmat_mean, softmax=False)
        return output, X, X_logits

class BagRepresentation_Histograms(nn.Module):
    def __init__(self, instance_representation, histnet):
        super(BagRepresentation_Histograms, self).__init__()
        self.instance_representation = instance_representation
        self.histnet = histnet

    def forward(self, X):
        X, X_logits = self.instance_representation(X)
        histogram = self.histnet(X)
        return histogram, X, X_logits

class BagRepresentation_Gaussians(nn.Module):
    def __init__(self, instance_representation, gaussian_layer):
        super().__init__()
        self.instance_representation = instance_representation
        self.gaussian_layer = gaussian_layer

    def forward(self, X):
        phi_X, X_logits = self.instance_representation(X)     
        Phi_X = self.gaussian_layer(phi_X)                    
        return Phi_X, phi_X, X_logits

class BagRepresentation_Mean(nn.Module):
    def __init__(self, instance_representation):
        super(BagRepresentation_Mean, self).__init__()
        self.instance_representation = instance_representation

    def forward(self, X):
        X, X_logits = self.instance_representation(X)
        mean = torch.mean(X, dim=0)
        return mean, X, X_logits

class Histogram_Layer(nn.Module):
    def __init__(self, input_size, n_bins=32):
        super(Histogram_Layer, self).__init__()
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

class Gaussian_Layer(nn.Module):
    def __init__(self, input_dim, gaussian_dim=5, n_gaussians=100, device='cpu'):
        super().__init__()
        self.proj = nn.Linear(input_dim, gaussian_dim)
        self.sigmoid = nn.Sigmoid()

        self.gm = GMLayer(n_features=gaussian_dim, num_gaussians=n_gaussians, device=device)
        geotorch.positive_definite(self.gm, "covariance")

        self.norm = nn.LayerNorm(n_gaussians, elementwise_affine=True)
        self.mean = MeanLayer()
        centers = self.gm.centers.detach().numpy()
        distances = scipy.spatial.distance.cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        cov = np.power(np.mean(np.min(distances, axis=1)) / 2, 2)
        cov = torch.eye(gaussian_dim).repeat(n_gaussians, 1, 1) * cov
        cov = cov.to(self.gm.covariance.device)
        self.gm.covariance.data.copy_(cov)


    def forward(self, x):
        z = self.sigmoid(self.proj(x))    
        lik = self.gm(z.unsqueeze(0))
     
        lik = self.norm(lik)               
        output = self.mean(lik)             
        return output.squeeze(0)


class Regressor_Solver(nn.Module):
    """
    This class defines a regressor, in the CovMat and also in de LMq and LZ. 
    Depending on the paramter softmax it apply or not this function (It is used only in the CovMat case.)
    """
    def __init__(self, input_dim, output_dim, dropout):
        super(Regressor_Solver, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(input_dim, input_dim//2))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("dropout_1", nn.Dropout(dropout))
        self.module.add_module("linear_2", nn.Linear(input_dim//2, output_dim))

    def forward(self, x, softmax):
        if softmax:
            return F.softmax(self.module(x), dim=-1)
        else:
            return self.module(x)

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

class EarlyStopping:
    def __init__(self, optimizer, patience=20, lr_end = 0.00001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.lr_end = lr_end
        self.optimizer = optimizer

    def __call__(self, val_loss, report_path, test_report=None):
        lr = self.optimizer.param_groups[0]['lr']

        if lr <= self.lr_end:
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = val_loss
            if test_report is not None:
                report = {
                    "mae_test": test_report["mae"].values.tolist(),
                    "mrae_test": test_report["mrae"].values.tolist(),
                    "val_loss": val_loss
                }
                with open(report_path, "w") as file:
                    file.write(json.dumps(report))
    
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0 
        else:
            self.best_score = val_loss
            self.counter = 0
            if test_report is not None:
                report = {
                    "mae_test": test_report["mae"].values.tolist(),
                    "mrae_test": test_report["mrae"].values.tolist(),
                    "val_loss": val_loss
                }
                with open(report_path, "w") as file:
                    file.write(json.dumps(report))            

class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

class StandardizedProtocol(AbstractProtocol):
    def __init__(self, protocol: AbstractProtocol, mean, std):
        self.protocol = protocol
        self.mean = mean.detach().cpu() if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
        self.std  = std.detach().cpu()  if isinstance(std, torch.Tensor)  else torch.tensor(std, dtype=torch.float32)
        self._cache = None 

    def total(self):
        return self.protocol.total()

    def __call__(self):
        if self._cache is not None:
            for item in self._cache:
                yield item
            return

        cache = []
        for sample, prev in self.protocol():
            X_tensor = torch.as_tensor(sample, dtype=torch.float32)
            X_st = ((X_tensor - self.mean) / self.std).numpy()
            cache.append((X_st, prev))
            yield (X_st, prev)

        self._cache = cache

def detach(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu()
    else:    
        return tensor.detach()

def standardize_lc(lc, ref=None, filename=None):
    print("Standardizing data")

    X = torch.tensor(ref.X, dtype=torch.float32)
    mean = X.mean(dim=0)
    std = X.std(dim=0)

    lcX = torch.tensor(lc.X, dtype=torch.float32)
    X_st = ((lcX - mean) / std).numpy()
    lc_st = LabelledCollection(X_st, lc.y)
    return lc_st, mean, std

def l2_loss(prev, M, q):
    prev = torch.as_tensor(prev, dtype=torch.float32, device=M.device)
    l2 = torch.norm(M@prev-q)
    return l2
 
def mq_loss(prev, M, q, regressor):
    """Regressor computing Mq."""
    prev = torch.as_tensor(prev, dtype=torch.float32, device=M.device)
    M_q = torch.cat((M.flatten(), q))
    p_hat = regressor(M_q, softmax=True) 
    ae = torch.mean(torch.abs(prev-p_hat))
    return ae

def triplet_loss(prev, num_classes, q, M, examples_prev_contr=50):
    """TRIPLET LOSS: min(|q - M@prev| - |q - M@prev_constr|)"""
    vect_matrix = qp.functional.uniform_prevalence_sampling(n_classes=num_classes, size=examples_prev_contr)
    p_aux = prev.reshape((1, -1))
    dist = distance.cdist(vect_matrix, p_aux, 'minkowski', p=1)
    ind = np.argmax(dist)
    
    dev = M.device
    prev_contr = torch.as_tensor(vect_matrix[ind], dtype=torch.float32, device=dev)
    prev = torch.as_tensor(prev, dtype=torch.float32, device=dev)
    
    eps = 1e-12
    M_norm = M / (torch.norm(M, dim=0, keepdim=True) + eps)
    q_norm = F.normalize(q, dim=-1)
    Mp = M_norm @ prev
    Mp_norm = F.normalize(Mp, dim=-1)

    Mp_c = M_norm @ prev_contr
    Mp_c_norm = F.normalize(Mp_c, dim=-1)

    positive_dist = F.pairwise_distance(q_norm, Mp_norm, p=2)
    negative_dist = F.pairwise_distance(q_norm, Mp_c_norm, p=2)

    loss_triplet = positive_dist - negative_dist #+ max(0, torch.linalg.norm(q)-1)

    return loss_triplet, prev_contr

def lZ_loss(regressor, Phi_X, prev):   
    """Direct Regressor (without computing Mq)."""             
    p_hat = regressor(Phi_X, softmax=True) 
    ae = torch.mean(torch.abs(prev-p_hat))
    return ae

def show_evaluation(train, test, bag_representation, solver, regressor, use_Mq, dataset_name, repeats=250, prefix=''):
    rep = RepresentationLearningQuantification(bag_representation.eval(), regressor, use_Mq, solver)
    if use_Mq:
        rep.fit(train)
    else:
        rep.fit(LabelledCollection(instances = [], labels=[]))
  
    if isinstance(test, AbstractProtocol):
        report = qp.evaluation.evaluation_report(rep, test, error_metrics=["mrae", "mae", "mkld"])
    
    else: #type LabelledCollection
        report = qp.evaluation.evaluation_report(rep, UPP(test, repeats=repeats), error_metrics=["mrae", "mae", "mkld"])
        
    RAE = float(np.mean(report["mrae"].values))
    AE = float(np.mean(report["mae"].values))
    KLD = float(np.mean(report["mkld"].values))
    print(f'\t{prefix} : {RAE=:.5f}\t{AE=:.5f}\t{KLD=:.5f}')
    return RAE, AE, KLD, report


def gen_M(data, bag_rep, chunk_size=512, train=True):
    M = []
    instance_rep_Xs = []
    ys = []
    Xs = []
    logits_Xs = []

    device = next(bag_rep.parameters()).device

    for i in data.classes_:
        sel = (data.y == i)
        Xi_np = data.X[sel]  

        Xi_parts = []
        for start in range(0, len(Xi_np), chunk_size):
            x_part = torch.tensor(Xi_np[start:start+chunk_size], dtype=torch.float32, device=device)
            Xi_parts.append(x_part)
        Xi = torch.cat(Xi_parts, dim=0)

        bag_rep_Xi, inst_rep_Xi, logs_Xi = bag_rep(Xi)


        if train:
            Xs.append(Xi)
            instance_rep_Xs.append(inst_rep_Xi)
            logits_Xs.append(logs_Xi)
            ys.append(torch.full((Xi.shape[0],), i, dtype=torch.long, device=device))

        M.append(bag_rep_Xi)

        del Xi_parts

    M = torch.stack(M).T

    if train:
        instance_rep_Xs = torch.cat(instance_rep_Xs)
        Xs = torch.cat(Xs)
        ys = torch.cat(ys)
        logits_Xs = None if logits_Xs[0] is None else torch.cat(logits_Xs)
        return M, instance_rep_Xs, Xs, ys, logits_Xs
    else:
        return M, None, None, None, None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training RLQuantification')
    parser.add_argument("-dataset", "--dataset", required=True)
    parser.add_argument("-d", "--device", help="Device cuda:0, cuda:1 or cpu", default="cpu")
    parser.add_argument("-dropout", "--dropout", type=float, default=0.0, help="Dropout rate (0.0 to 1.0)")
    parser.add_argument("-L", "--L", help= "Representation loss. It can be: L2, LT, LMq or LZ", default="L2")
    parser.add_argument("-Phi", "--Phi", help= "Bag representation. It can be: Phi_H, Phi_mu, Phi_cm, Phi_G", default="Phi_mu")
    parser.add_argument("-phi", "--phi", help= "Instance representation. It can be: phi_h, phi_l, phi_f, phi_A, phi_cnn or phi_bert", default="phi_h")
    parser.add_argument("-s", "--seed", type=int, help = "Seed used, default 2032", default=2032)
    parser.add_argument("-error", "--error", help = "Optimization error used. It can be: AE, RAE or KLD. Default AE", default="AE")
    parser.add_argument("-lr", "--lr", type=float, default=0.0005, help = "Starting learning rate.")
    parser.add_argument("-lr_end", "--lr_end", type=float, default=0.0001, help = "Final learning rate.")
    parser.add_argument("-b", "--n_bags", type=int, default=1000, help = "Number of bags train.")
    
    return parser.parse_args()

def load_dataset(dataset_name):
    # LeQua2022 
    if dataset_name in ["T1A", "T1B", "T2A", "T2B"]:
        qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[dataset_name]  
        training, val_gen, test_gen = fetch_lequa2022(task=dataset_name, data_home="/media/nas/olayap/env_olaya/Doctorado/data/")
        train, mean, std = standardize_lc(training,training,filename="mean_std_{}.pth".format(dataset_name))
        test_std = StandardizedProtocol(test_gen, mean, std)
        val_std = StandardizedProtocol(val_gen, mean, std)
        return train, test_std, val_std, train.n_classes
    
    # LeQua2024
    elif dataset_name in ["T1", "T2", "T3"]:
        qp.environ['SAMPLE_SIZE'] = LEQUA2024_SAMPLE_SIZE[dataset_name]  
        training, val_gen, test_gen = fetch_lequa2024(task=dataset_name, data_home="/media/nas/olayap/env_olaya/Doctorado/data/")
        
        train, mean, std = standardize_lc(training,training,filename="mean_std_{}.pth".format(dataset_name))
        test_std = StandardizedProtocol(test_gen, mean, std)
        val_std = StandardizedProtocol(val_gen, mean, std)
        return train, test_std, val_std, train.n_classes
    
    # CIFAR
    elif dataset_name == "CIFAR10":
        transform = transforms.ToTensor()
        train_ds = CIFAR10(root='/media/nas/olayap/env_olaya/Doctorado/data', train=True, download=True, transform=transform)
        test_ds  = CIFAR10(root='/media/nas/olayap/env_olaya/Doctorado/data', train=False, download=True, transform=transform)

        x_train = torch.stack([img for img, _ in train_ds]) #.numpy() #.view(len(train_ds), -1).numpy()
        y_train = np.array([label for _, label in train_ds])
        x_test = torch.stack([img for img, _ in test_ds]) #.numpy() #.view(len(test_ds), -1).numpy()
        y_test = np.array([label for _, label in test_ds])

        train = LabelledCollection(x_train, y_train)
        train, val = train.split_stratified(train_prop=0.8)
        test  = LabelledCollection(x_test, y_test)
        n_classes = train.n_classes

        return train, test, val, n_classes
    
    elif dataset_name == "CIFAR100coarse":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_ds = CIFAR100Coarse(root='/media/nas/olayap/env_olaya/Doctorado/data', train=True, download=True, transform=transform_train)
        test_ds  = CIFAR100Coarse(root='/media/nas/olayap/env_olaya/Doctorado/data', train=False, download=True, transform=transform_test)

        x_train = torch.stack([img for img, _ in train_ds]) #.numpy() #.view(len(train_ds), -1).numpy()
        y_train = np.array([label for _, label in train_ds])
        x_test = torch.stack([img for img, _ in test_ds]) #.numpy() #.view(len(test_ds), -1).numpy()
        y_test = np.array([label for _, label in test_ds])

        train = LabelledCollection(x_train, y_train)
        train, val = train.split_stratified(train_prop=0.8)
        test  = LabelledCollection(x_test, y_test)
        n_classes = train.n_classes

        return train, test, val, n_classes
    
    # UCI 
    else:
        if dataset_name == "connect-4":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, max_train_instances=47290)
        elif dataset_name == "chess":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=251)
        elif dataset_name in ["dry-bean", "hand_digits"]:
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
        elif dataset_name == "shuttle":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=200, max_train_instances=40429)
        elif dataset_name == "poker_hand":
            data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name, min_class_support=4100, max_train_instances=800000)
            
        n_classes = data.n_classes
        qp.data.preprocessing.standardize(data, inplace=True)
        train, test = data.train_test
        train, val = train.split_stratified(train_prop=0.8)
    
        return train, test, val, n_classes

def initialize_model_components(args, train, patience=20):
    num_classes = len(train.classes_)
    #CAMBIO
    n_feat = train.X.shape[1] if args.phi != "phi_bert" else None
    
    dropout = args.dropout  
    lr=args.lr
    lr_end=args.lr_end

    phis = ["phi_h", "phi_l", "phi_f", "phi_A", "phi_bert"]
    Phis = ["Phi_H", "Phi_mu", "Phi_cm", "Phi_G"]
    losses = ["L2", "LT", "LMq", "LZ"]
    params = []
    
    #INSTANCE REPRESENTATION
    if args.phi in phis:
        if args.phi == "phi_h":
            if args.dataset in IMAGES:
                in_channels = train.X.shape[1]
                img_size = train.X.shape[2]
                n_feat = 32
                instance_representation = InstanceRepresentationImages(in_channels, n_feat, return_next_to_last=False, num_classes=num_classes, img_size=img_size)
            else:
                instance_representation = InstanceRepresentation(n_feat, dropout, return_next_to_last=False, num_classes=num_classes)

        elif args.phi == "phi_l":
            if args.dataset in IMAGES:
                in_channels = train.X.shape[1]
                img_size = train.X.shape[2]
                n_feat = 32
                instance_representation = InstanceRepresentationImages(in_channels, n_feat, return_next_to_last=True, num_classes=num_classes, img_size=img_size)
            else:
                instance_representation = InstanceRepresentation(n_feat, dropout, return_next_to_last=True)
            latent_dims = instance_representation.dimensions()
            linear_classif = nn.Linear(latent_dims, num_classes)
            params += list(linear_classif.parameters())
        
        elif args.phi == "phi_A":
            if args.dataset in IMAGES:
                in_channels = train.X.shape[1]
                img_size = train.X.shape[2]
                n_feat = 32
                instance_representation = InstanceRepresentationImages(in_channels, n_feat, return_next_to_last=True, num_classes=num_classes, img_size=img_size)
                latent_dims = instance_representation.dimensions() 
                instance_representation_decoder = InstanceRepresentationImages_decoder(latent_dims, in_channels)
            else:
                instance_representation = InstanceRepresentation(n_feat, dropout, return_next_to_last=True)
                latent_dims = instance_representation.dimensions()
                instance_representation_decoder = InstanceRepresentation_decoder(n_feat, latent_dims, dropout)
            params += list(instance_representation_decoder.parameters())
        
        elif args.phi == "phi_f":
            if args.dataset in IMAGES:
                in_channels = train.X.shape[1]
                img_size = train.X.shape[2]
                n_feat = 32
                instance_representation = InstanceRepresentationImages(in_channels, n_feat, return_next_to_last=True, num_classes=num_classes, img_size=img_size)
            else:
                instance_representation = InstanceRepresentation(n_feat, dropout, return_next_to_last=True)
        
        elif args.phi == "phi_bert":
            instance_representation = phi_bert(output_size=256)
    
    else:
        print("ERROR. The instance representation is wrong. It could be: phi_h, phi_l, phi_f, phi_A or phi_bert")
    
    latent_dims = instance_representation.dimensions()

    
    #BAG REPRESENTATION
    if args.Phi in Phis:
        if args.Phi == "Phi_mu":
            bagRepresentation = BagRepresentation_Mean(instance_representation)
            params += list(bagRepresentation.parameters())
        
        elif args.Phi == "Phi_cm":
            regressor_cm = Regressor_Solver((latent_dims*(latent_dims+1)//2)+latent_dims, latent_dims, dropout)
            bagRepresentation = BagRepresentation_CovMat(instance_representation, regressor_cm)
            params += list(bagRepresentation.parameters())
            
        elif args.Phi == "Phi_H":
            n_bins = 16
            histnet = Histogram_Layer(input_size = latent_dims, n_bins = n_bins)
            bagRepresentation = BagRepresentation_Histograms(instance_representation, histnet)
            params += list(bagRepresentation.parameters()) 
        
        elif args.Phi == "Phi_G":
            n_gaussians = 100
            gaussian_dimensions = 5
            gmnet = Gaussian_Layer(latent_dims, gaussian_dimensions, n_gaussians, device)
            bagRepresentation = BagRepresentation_Gaussians(instance_representation, gmnet)
            params += list(bagRepresentation.parameters())         
    else:
        print("ERROR. The bag representation is wrong. It could be: Phi_H, Phi_mu, Phi_cm or Phi_G")

    #REPRESENTATION LOSS
    if args.L in losses:
        if args.L == "L2":
            solver= "fro"
            use_Mq = True

        elif args.L == "LT":
            solver= "fro"
            use_Mq = True

        elif args.L == "LMq":
            solver= "regressor"
            use_Mq = True
            if args.Phi == "Phi_H":
                regressor = Regressor_Solver(input_dim=(latent_dims*num_classes+latent_dims)*n_bins, output_dim=num_classes, dropout=dropout)
            elif args.Phi == "Phi_G":
                regressor = Regressor_Solver(input_dim=n_gaussians*num_classes+n_gaussians, output_dim=num_classes, dropout=dropout)
            else:
                regressor = Regressor_Solver(input_dim=latent_dims*num_classes+latent_dims, output_dim=num_classes, dropout=dropout)
            
            params += list(regressor.parameters())

        elif args.L == "LZ":
            solver= "regressor"
            use_Mq = False
            if args.Phi == "Phi_H":
                regressor = Regressor_Solver(input_dim=latent_dims*n_bins, output_dim=num_classes, dropout=dropout)
            elif args.Phi == "Phi_G":
                regressor = Regressor_Solver(input_dim=n_gaussians, output_dim=num_classes, dropout=dropout)
            else:
                regressor = Regressor_Solver(input_dim=latent_dims, output_dim=num_classes, dropout=dropout)
            
            params += list(regressor.parameters())      
    else:
        print("ERROR. The representation loss is wrong. It could be: L2, LT, LMq or LZ")
    
    instance_representation = instance_representation.to(device)
    bagRepresentation = bagRepresentation.to(device)

    regressor_cm = regressor_cm.to(device) if 'regressor_cm' in locals() else None
    histnet = histnet.to(device) if 'histnet' in locals() else None
    gmnet = gmnet.to(device) if 'gmnet' in locals() else None
    regressor = regressor.to(device) if 'regressor' in locals() else None
    linear_classif = linear_classif.to(device) if 'linear_classif' in locals() else None
    instance_representation_decoder = instance_representation_decoder.to(device) if 'instance_representation_decoder' in locals() else None

    optimizer = optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5, cooldown=0, verbose=True)
    early_stopping = EarlyStopping(optimizer=optimizer, patience=patience, lr_end=lr_end)
    return optimizer, scheduler, early_stopping, instance_representation, bagRepresentation, regressor_cm, histnet, gmnet, regressor, linear_classif, instance_representation_decoder, use_Mq, solver

def train_model(args, train, val, test, optimizer, scheduler, early_stopping, instance_representation, bag_representation, regressor_cm, histnet, gmnet, regressor, linear_classif, instance_representation_decoder, use_Mq, solver):
    
    max_sample_size = 250
    min_sample_size = 100
    n = 20_000
    if args.dataset in ["T1A", "T1B", "T2A", "T2B"]:
        qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[args.dataset]
        sample_size = LEQUA2022_SAMPLE_SIZE[args.dataset]
    elif args.dataset in ["T1", "T2", "T3"]:
        qp.environ['SAMPLE_SIZE'] = LEQUA2024_SAMPLE_SIZE[args.dataset]
        sample_size = LEQUA2024_SAMPLE_SIZE[args.dataset]
    elif args.dataset in ["CIFAR10", "CIFAR100coarse"]:
        qp.environ['SAMPLE_SIZE'] = 250
        n = 15_000
        max_sample_size = 500
        min_sample_size = 200
    else:
        qp.environ['SAMPLE_SIZE'] = 250
        max_sample_size = 500

    n_epochs = 1000
    min_val_loss = np.inf
    loss_str = LossStr()   
    mse_loss = nn.MSELoss()
    cr_en_loss = nn.CrossEntropyLoss()
    report_dir = os.path.join(f"results/reports_{args.seed}", args.dataset)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(f"results/reports_{args.seed}", args.dataset, args.phi+'_'+args.Phi+'_'+args.L+'_'+args.error+('_dropout' if args.dropout != 0.0 else ''))
    num_classes = train.n_classes
    HARD_PATIENCE = 40
    epochs_since_best = 0

    for epoch in range(n_epochs):
        
        if args.dataset not in ["T1A", "T1B", "T2A", "T2B", "T2", "T3"]:
            sample_size = random.randint(min_sample_size, max_sample_size)

        if use_Mq:
            LM, Lq = train.split_stratified(train_prop=0.75)
            
            if len(LM) > n:
                LM = LM.sampling(n, *LM.prevalence())

            upp_gen = UPP(Lq, sample_size=sample_size, random_state=None, return_type="sample_prev", repeats=args.n_bags)
        else:
            upp_gen = UPP(train, sample_size=sample_size, random_state=None, return_type="labelled_collection", repeats=args.n_bags)

        for i, data in enumerate(upp_gen()):
            instance_representation.train()
            bag_representation.train()

            if args.phi == "phi_A":
                instance_representation_decoder.train()
            if args.Phi == "Phi_cm":
                regressor_cm.train()
            if args.Phi == "Phi_H":
                histnet.train()
            if args.Phi == "Phi_G":
                gmnet.train()
            if args.L == "LMq" or args.L == "LZ":
                regressor.train()

            if use_Mq:
                sam, prev = data
                M, phi_X, X, y, X_logits  = gen_M(LM, bag_representation, chunk_size=512, train=True)
                dev = next(bag_representation.parameters()).device
                sam = torch.tensor(sam, dtype=torch.float32, device=dev)
                q, Xq, logits = bag_representation(sam)

            else:
                dev = next(bag_representation.parameters()).device
                X = torch.tensor(data.X, dtype=torch.float32, device=dev)
                prev = data.prevalence()
                prev = torch.as_tensor(prev, dtype=torch.float32, device=dev)
                y = torch.tensor(data.y, dtype=torch.long, device=dev)
                Phi_X, phi_X, X_logits = bag_representation(X)

            # LOSSES
            loss = 0 
            if use_Mq:
                if args.L == "L2":
                    l2 = l2_loss(prev, M, q)
                    loss_str.add(l2, 'l2_loss')
                    loss += l2

                elif args.L == "LMq":
                    ae = mq_loss(prev, M, q, regressor)
                    loss_str.add(ae, 'LMq')
                    loss += ae

                elif args.L == "LT":
                    triplet, p_c = triplet_loss(prev, num_classes, q, M, examples_prev_contr=50)
                    loss_str.add(triplet, 'LT')
                    loss += triplet
            else:
                ae = lZ_loss(regressor, Phi_X, prev)
                loss_str.add(ae, 'LZ')
                loss += ae 

            if args.phi == "phi_A" :
                phi_X_hat = instance_representation_decoder(phi_X)
                loss_autoencoder = mse_loss(phi_X_hat, X)
                loss_str.add(loss_autoencoder, 'phi_A')
                loss += loss_autoencoder

            if args.phi == "phi_l" :  
                X = linear_classif(phi_X)
                cr_loss = cr_en_loss(X, y)
                loss_str.add(cr_loss, 'phi_l')
                loss += cr_loss

            if args.phi == "phi_h" :
                cr_loss = cr_en_loss(X_logits, y)
                loss_str.add(cr_loss, 'phi_h')
                loss += cr_loss

            loss_str.add(loss, 'total')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss_str}')
        
        #training
        if use_Mq:     
            show_evaluation(LM, Lq, bag_representation, solver, regressor, use_Mq, args.dataset, prefix='Training')
        else:
            show_evaluation([], train, bag_representation, solver, regressor, use_Mq, args.dataset, prefix='Training')
        
        #validation   
        val_rae_loss, val_ae_loss, val_kld_loss, _ = show_evaluation(train, val, bag_representation, solver, regressor, use_Mq, args.dataset, prefix='Validation')
        
        #test
        if args.error == "RAE":
            val_loss = val_rae_loss
        elif args.error == "AE":
            val_loss = val_ae_loss
        elif args.error == "KLD":
            val_loss = val_kld_loss
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_since_best = 0

            if isinstance(val, AbstractProtocol): 
                _, _, _, test_report = show_evaluation(train, test, bag_representation, solver, regressor, use_Mq, args.dataset, prefix='Test')
            else:
                _, _, _, test_report = show_evaluation(train+val, test, bag_representation, solver, regressor, use_Mq, args.dataset, prefix='Test')
        
        else:
            epochs_since_best += 1

        if args.error == "RAE":
            scheduler.step(val_rae_loss)
            early_stopping(val_rae_loss, report_path, test_report)
        elif args.error == "AE":
            scheduler.step(val_ae_loss)
            early_stopping(val_ae_loss, report_path, test_report)
        elif args.error == "KLD":
            scheduler.step(val_kld_loss)
            early_stopping(val_kld_loss, report_path, test_report)

        if USE_WANDB:
            logs = {"train_loss": loss.item(), "val_RAE_loss": val_rae_loss, "val_AE_loss": val_ae_loss, "val_KLD_loss": val_kld_loss, "lr": optimizer.param_groups[0]['lr']}
            for loss_name, loss_values in loss_str.losses.items():
                logs[f"{loss_name}"] = np.mean(loss_values)

            wandb.log(logs, step=epoch)

        if epochs_since_best >= HARD_PATIENCE:
            print(f"Early stopping. {HARD_PATIENCE} epochs without improvement.")
            break

        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device(args.device)
    TOCUDA = (device.type == "cuda")
        
    #WANDB
    if USE_WANDB:
        wandb.login()
        wandb_project_name = f"RLQ_{args.seed}"
        wandb_execution_name = args.dataset+'_'+args.phi+'_'+args.Phi+'_'+args.L+'_'+args.error+('_dropout' if args.dropout != 0.0 else '')
        wandb.init(
            project=wandb_project_name,
            name=wandb_execution_name,
            save_code=True,
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    train, test, val, num_classes = load_dataset(args.dataset)
    
    optimizer, scheduler, early_stopping, instance_representation, bagRepresentation, regressor_cm, histnet, gmnet, regressor, linear_classif, instance_representation_decoder, use_Mq, solver = initialize_model_components(args, train, patience=20)
        
    train_model(args, train, val, test, optimizer, scheduler, early_stopping, instance_representation, bagRepresentation, regressor_cm, histnet, gmnet, regressor, linear_classif, instance_representation_decoder, use_Mq, solver)

