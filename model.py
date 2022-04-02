import torch 
from torch import nn
import numpy as np
from layer import graph_convolution_layer

class GCN(nn.Module):

    def __init__(self, input_dim, hidden1_dim):
        super(GCN, self).__init__()
        self.hid1 = input_dim * hidden1_dim
        self.conv1 = graph_convolution_layer(input_dim, hidden1_dim)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hid1, 16)
        self.linear2 = nn.Linear(16, 1)
        self.softmax = nn.Softmax()
        


    def forward(self, A, X):
        H = self.conv1(A, X)
        H = self.relu(H)

        F = H.flatten()
        l = self.hid1  - F.shape[0]
        if l != 0:
            P = nn.ConstantPad1d(padding=(0, l), value=0)
            F = P(F)

        Y = self.linear1(F)
        Y = self.relu(Y)
        Y = self.linear2(Y)
        y = self.relu(Y)

        return y
    
    def loss(y_pred, y):
        return nn.CrossEntropyLoss(y_pred, y)

class GCN_2(nn.Module):

    def __init__(self, shape):
        super(GCN_2, self).__init__()
        self.input_dim = shape[0]
        self.hidden1_dim = shape[1]
        self.hidden2_dim = shape[2] 
        self.hid = self.hidden1_dim * self.hidden2_dim
        self.conv1 = graph_convolution_layer(self.input_dim, self.hidden1_dim)
        self.relu = nn.ReLU()
        self.conv2 = graph_convolution_layer(self.hidden1_dim, self.hidden2_dim)
        self.linear1 = nn.Linear(self.hid, 1)
        self.softmax = nn.Softmax()
        


    def forward(self, A, X):
        H = self.conv1(A, X)
        H = self.relu(H)
        H = self.conv2(A, H)

        F = H.flatten()
        
        l = self.hid  - F.shape[0]
        if l != 0:
            P = nn.ConstantPad1d(padding=(0, l), value=0)
            F = P(F)
        #F = H[0]

        Y = self.linear1(F)
        y = self.relu(Y)

        return y
    
    def loss(y_pred, y):
        return nn.CrossEntropyLoss(y_pred, y)



    
