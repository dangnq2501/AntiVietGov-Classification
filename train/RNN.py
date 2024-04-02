#------------------------loading data------------------------------------------
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

trainloader =  joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/trainloader_300_drop')
testloader = joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/testloader_300_drop')

BATCH_SIZE = 128
train_dataloader = torch.utils.data.DataLoader(trainloader.dataset,batch_size= BATCH_SIZE, shuffle= True)
test_dataloader = torch.utils.data.DataLoader(testloader.dataset,batch_size= BATCH_SIZE)



#--------------------------------------------training loop----------------------
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

def train_step(model : nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_function : nn.Module,
               optimizer,
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

    print('------------------Train Result----------------------------')
    print(f'Training loss : {loss} | F1_score : {f1_score(all_y_true,all_y_pred)}')
    print(f'Confusion matrix :')
    print(confusion_matrix(all_y_true,all_y_pred))

def test_step(model : nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_function : nn.Module,
              optimizer,
              device = 'cuda'):

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

    print('------------------Test Result----------------------------')
    print(f'Testing loss : {loss} | F1_score : {f1_score(all_y_true,all_y_pred)}')
    print(f'Confusion matrix :')
    print(confusion_matrix(all_y_true,all_y_pred))
    print('---------------------------------------------------------')

matrix_size = (300,300)


class RNNmodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(input_size = matrix_size[1],hidden_size = 32,
            num_layers = 3, batch_first = True, bidirectional = True)

        self.fc = nn.LazyLinear(out_features = 1)

    #it output [0,1,2,....,seq_length - 1]
    #just take the last array element in case of classification or anything like that
    def forward(self, X, state=None):
        rnn_outputs, _ = self.rnn(X, state)

        return self.fc(rnn_outputs[:,-1,:])

device = 'cuda'
model = RNNmodel().to(device)
BCE_loss = nn.BCEWithLogitsLoss()
Adam_optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.001)


epochs = 60

for epoch in range(0,epochs):
    print(f'Epoch {epoch}=======================================')
    train_step(model,train_dataloader,BCE_loss,Adam_optimizer,device = device)
    test_step(model,test_dataloader,BCE_loss,Adam_optimizer,device = device)
    #break