import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quapy as qp
from quapy.protocol import UPP
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PhiPequena(nn.Module):
    def __init__(self, X_shape, n_classes):
        super(PhiPequena, self).__init__()
        self.module = torch.nn.Sequential()
        self.module.add_module("linear_1", nn.Linear(X_shape, 64))
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("dropout_1", nn.Dropout(0.5))
        self.module.add_module("linear_2", nn.Linear(64, 10))
        self.module.add_module("relu_2", nn.ReLU())
        #self.module.add_module("linear_3", nn.Linear(10, n_classes))

    def forward(self, x):
        return self.module(x)

class PhiGrande(nn.Module):
    def __init__(self, phi_pequena, num_classes):
        super(PhiGrande, self).__init__()
        self.phi_pequena = phi_pequena
        self.num_classes = num_classes
    
    def forward(self, X):
        X = self.phi_pequena(X)
        return torch.mean(X, dim=0), X

def cosine_distance(x1, eps=1e-8):
    w1 = x1.norm(p=2, dim=0, keepdim=True)
    return (x1.t() @ x1) / (w1.t() * w1).clamp(min=eps)

def regularization_classif(X, y, linear_clas):
    X = linear_clas(X)
    cr_en_loss = nn.CrossEntropyLoss()
    loss = cr_en_loss(X, y)
    return loss



if __name__ == '__main__':
    train, test = qp.datasets.fetch_UCIMulticlassDataset('dry-bean').train_test
    num_classes = len(train.classes_)
    test_dataset = TensorDataset(torch.tensor(test.X, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, shuffle=False)


    def gen_M(data, phi_grande):
        M = []
        X = []
        y = []
        for i in data.classes_:
            phi_gr, phi_peqs = phi_grande(torch.tensor(data.X[data.y==i], dtype=torch.float32))
            M.append(phi_gr)
            X.extend(phi_peqs)
            y.append(torch.full(size=(np.sum(data.y == i),), fill_value= i))
        #M = [phi_grande(torch.tensor(data.X[data.y==i], dtype=torch.float32))  for i in data.classes_]
        M = torch.stack(M).T 
        X = torch.stack(X)
        y = torch.concat(y)
        return M, X, y

    n_feat = train.X.shape[1]
    phi_pequena = PhiPequena(n_feat, num_classes)
    linear_classif = nn.Linear(10, num_classes)
    params = list(phi_pequena.parameters()) + list(linear_classif.parameters())

    optimizer = optim.Adam(params, lr=0.001)

    n_ages = 10
    for age in range(n_ages):
        LM, Lq = train.split_stratified(train_prop=0.5)
        X_LM, y_LM = torch.tensor(LM.X, dtype=torch.float32), torch.tensor(LM.y, dtype=torch.int)
        batch_size = 64 

        n_epochs = 10
        for epoch in range(n_epochs):
            upp_gen = UPP(Lq, sample_size=100, random_state=None, return_type="sample_prev")
            for i, (sam, prev) in enumerate(upp_gen()):
                phi_pequena.train()
                phi_grande = PhiGrande(phi_pequena, num_classes)
                phi_grande.train()
                M, X, y = gen_M(LM, phi_grande)
                sam = torch.tensor(sam, dtype=torch.float32)
                prev = torch.tensor(prev, dtype=torch.float32)
                q, _ = phi_grande(sam)
                
                loss_reconstruction =  torch.norm(M@prev - q) -torch.norm(cosine_distance(M))
                loss_classif = regularization_classif(X, y, linear_classif)
                loss = loss_classif+loss_reconstruction
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Age {age+1}/Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
            #print("M", M)
            #print("q", q)
            #print("cd", cosine_distance(M))
