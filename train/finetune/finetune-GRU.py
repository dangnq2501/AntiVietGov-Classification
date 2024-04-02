from torch import nn
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from torch.autograd import Variable
import gensim, logging
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
from pathlib import Path

trainloader =  joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/trainloader_300_drop')
testloader = joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/testloader_300_drop')

BATCH_SIZE = 64
train_dataloader = torch.utils.data.DataLoader(trainloader.dataset,batch_size= BATCH_SIZE, shuffle= True)
test_dataloader = torch.utils.data.DataLoader(testloader.dataset,batch_size= BATCH_SIZE)

#we will output F1 score or confusion matrix at each step

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def F1_tensor(y_true, y_pred):
    y_true = y_true.to('cpu').numpy()
    y_pred = y_pred.to('cpu').numpy()
    return f1_score(y_true, y_pred)

def Confusion_matrix_tensor(y_true, y_pred):
    y_true = y_true.to('cpu').numpy()
    y_pred = y_pred.to('cpu').numpy()
    return f1_score(y_true, y_pred)

def convert_from_tensor(y): #convert from tensor to some kind of array that we can use numpy
    return y.cpu().detach().numpy().reshape(-1)

def take_all_elem(container, target):
    for x in target:
        if (x != 0 and x != 1):
            container.append(1)
        else:
            container.append(x)

def save_model(model):
    MODEL_PATH = Path('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/models')
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = 'best_GRUmodel.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f'Update new best model to : {MODEL_SAVE_PATH}')
    torch.save(obj = model.state_dict(),f = MODEL_SAVE_PATH)

def train_step(model : nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_function : nn.Module,
               optimizer,
               keep_log = True,
               device = 'cuda'):
    model.train()
    loss = 0

    all_y_true = []
    all_y_pred = []

    for (X_train,y_train) in data_loader:
        X_train = X_train.to(device)
        y_train = y_train.unsqueeze(1).to(device)

        y_pred = model(X_train)
        y_pred01 = torch.round(torch.sigmoid(y_pred))

        batch_loss = loss_function(y_pred,y_train)
        loss += batch_loss

        take_all_elem(all_y_true,convert_from_tensor(y_train))
        take_all_elem(all_y_pred,convert_from_tensor(y_pred01))


        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    loss /= len(data_loader)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    #print(all_y_true)
    #print(np.unique(all_y_true))
    if (keep_log == True):
        print('------------------Train Result----------------------------')
        print(f'Training loss : {loss} | F1_score : {f1_score(all_y_true,all_y_pred)}')
        print(f'Confusion matrix :')
        print(confusion_matrix(all_y_true,all_y_pred))

best_f1_score = -1
def test_step(model : nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_function : nn.Module,
              keep_log = True,
              device = 'cuda',):

    model.eval()
    loss,acc = 0,0

    all_y_true = []
    all_y_pred = []

    with torch.inference_mode():
        loss = 0

        for (X_test,y_test) in data_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            test_logits = model(X_test).squeeze()
            test_01 = torch.round(torch.sigmoid(test_logits))

            batch_loss = loss_function(test_logits,y_test)

            loss += batch_loss

            take_all_elem(all_y_true,convert_from_tensor(y_test))
            take_all_elem(all_y_pred,convert_from_tensor(test_01))


        loss /= len(data_loader)
        acc /= len(data_loader)

    current_f1_score = f1_score(all_y_true,all_y_pred)
    if (keep_log == True):
        print('------------------Test Result----------------------------')
        print(f'Testing loss : {loss} | F1_score : {current_f1_score}')
        print(f'Confusion matrix :')
        print(confusion_matrix(all_y_true,all_y_pred))
        print('---------------------------------------------------------')
    
    return current_f1_score

matrix_size = (300,300)

class GRUmodel(nn.Module):
    def __init__(self,hidden_size = 32,num_layers = 3, bidirectional = False):
        super().__init__()

        self.rnn = nn.GRU(input_size = matrix_size[1],hidden_size = hidden_size,
            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)

        self.fc = nn.LazyLinear(out_features = 1)

    #it output [0,1,2,....,seq_length - 1]
    #just take the last array element in case of classification or anything like that
    def forward(self, X, state=None):
        rnn_outputs, _ = self.rnn(X, state)

        return self.fc(rnn_outputs[:,-1,:])
    
    def feature_extract(self, X, state = None):
        rnn_outputs, _ = self.rnn(X, state)
        return rnn_outputs[:,-1,:]

device = 'cuda'
model = GRUmodel().to(device)
BCE_loss = nn.BCEWithLogitsLoss()
Adam_optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.001)

import optuna

def optuna_training(model : nn.Module,
               loss_function : nn.Module,
               optimizer_string,
               n_epoch,
               lr,
               device = 'cuda'):

    optimizer = torch.optim.Adam(params = model.parameters(),lr = lr)
    if (optimizer_string == 'AdamW'):
        optimizer = torch.optim.AdamW(params = model.parameters(),lr = lr)

    print(f'###################Another Try########################')
    best_f1_score = -1

    keep_log = False

    for epoch in range(n_epoch):
        if (keep_log == True):
            print(f'Epoch {epoch}=======================================')
        train_step(model,train_dataloader,loss_function,optimizer,keep_log = keep_log)
        model_f1_score = test_step(model,test_dataloader,loss_function,keep_log = keep_log)
        best_f1_score = max(best_f1_score,model_f1_score)
    
    torch.cuda.empty_cache()
    
    print(f'Trial best f1 score {best_f1_score}')
    return 1 - best_f1_score

def objective(trial):
    n_epochs = trial.suggest_int("n_epochs", 30, 50)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 512)
    num_layer = trial.suggest_int("num_layer", 2, 6)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    optimizer_string = trial.suggest_categorical('optimizer_string', ['Adam', 'AdamW'])
    model = GRUmodel(hidden_size = hidden_size, num_layers = num_layer, 
                   bidirectional = bidirectional).to(device)
    
    return optuna_training(model,BCE_loss,optimizer_string,n_epochs,lr)

study = optuna.create_study()
study.optimize(objective, n_trials = 50)

trial = study.best_trial

print("Best F1 score: {}".format(1 - trial.value))
print("Best hyperparameters: {}".format(trial.params))