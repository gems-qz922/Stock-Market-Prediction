import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.ticker as ticker
from livelossplot import PlotLosses

class StockDataset(TensorDataset):
  def __init__(self,data,seq_len,norm_base,target_len):
    self.norm_base = norm_base
    self.data = pd.Series([data.iloc[0] for i in range(seq_len)]+list(data.values)) # make it as pd series, because want to use builtin rolling window
    windows = self.data.rolling(window=(seq_len+target_len)) # rolling window
    self.x, self.y = torch.zeros((len(data),1,seq_len)), torch.zeros((len(data),1,target_len)) # create the x, y as input and target respectively
    # x.shape --> (total,1,30); y.shape --> (total,1,1)
    count = 0
    for w in windows:
      w = w.values
      if len(w) != seq_len+target_len:
        continue
      else: # only record window that is exact length with seq_len + target_len --> can be seperate as x, y
        target = w[-int(target_len):]
        self.x[count,0] = torch.tensor(w[:seq_len], dtype=torch.float32)
        self.y[count,0] = torch.tensor(target, dtype=torch.float32)
        count += 1
    self.seq_len = seq_len
    self.target_len = target_len

  def norm(self,input):
    return input/self.norm_base

  def __getitem__(self,idx):
    return self.norm(self.x[idx]),self.norm(self.y[idx])

  def __len__(self):
    return len(self.x)



class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # we will streamline the implementation of the LSTM by combining the
        # weights for all 4 operations (input gate, forget gate, output gate, candidate update)
        self.i2h = nn.Linear(input_size, hidden_size * 4, bias=bias) # create a linear layer to map from input to hidden space
        self.h2h = nn.Linear(hidden_size, hidden_size * 4, bias=bias) # create a linear layer to map from previous to current hidden space
        self.reset_parameters() # initialise the parameters

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, h, c):
        gates = self.i2h(input) + self.h2h(h) # apply the weights to both input and previous state
        input_gate, forget_gate, candidate_update, output_gate = gates.chunk(4, 1)  # separate the output into each of the LSTM operations
        # apply the corresponding activations
        i_t = torch.sigmoid(input_gate)                                    
        f_t = torch.sigmoid(forget_gate)
        c_t = torch.tanh(candidate_update)
        o_t = torch.sigmoid(output_gate)
        c = f_t * c + i_t * c_t # calculate the next cell state
        h = o_t * torch.tanh(c) # calculate the next hidden state
        return h, c



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bias=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.rnn_cell_list = nn.ModuleList() # create a list of modules
        # create each layer in the network
        # take care when defining the input size of the first vs later layers
        for l in range(self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.input_size if l == 0 else self.hidden_size,
                                               self.hidden_size,
                                               self.bias))
        self.h2o = nn.Linear(self.hidden_size, self.output_size) # create a final linear layer from hidden state to network output

    def init_hidden(self,  batch_size=1):
        # initialise the hidden state and cell state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False))

    def forward(self, input, h0, c0):
        # Input of shape (batch_size, seqence length , input_size)
        # Output of shape (batch_size, output_size)
        outs = []
        hidden = []
        cell = []
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])
            cell.append(c0[layer, :, :])
        # iterate over all elements in the sequence
        for t in range(input.size(1)):
            # iterate over each layer
            for layer in range(self.num_layers):
                # apply each layer
                # take care to apply the layer to the input or the
                # previous hidden state depending on the layer number
                if layer == 0:
                    hidden_l, cell_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer], cell[layer])
                else:
                    hidden_l, cell_l = self.rnn_cell_list[layer](hidden[layer-1], hidden[layer], cell[layer])
                # store the hidden and cell state of each layer
                hidden[layer] = hidden_l
                cell[layer] = cell_l
            # the hidden state of the last layer needs to be recorded
            # to be used in the output
            outs.append(hidden_l)
        # calculate output for each element in the sequence
        out = torch.stack([self.h2o(out) for out in outs], dim=1)

        return out



class LSTM_GEN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_GEN, self).__init__()
        # define your layers and activations
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = LSTM(self.input_size, self.hidden_size, self.num_layers, self.output_size) # add out LSTM

    def forward(self, x):
        batch_size = x.size(0)
        state_h, state_c = self.lstm.init_hidden(batch_size)  # initialise hidden state
        output = self.lstm(x, state_h, state_c)  # apply the LSTM
        return output


def train_lstm_gen(model, optimizer, criterion, dataloader):
    model.train()                       # set model to train mode
    train_loss, train_accuracy = 0, 0   # initialise the loss 
    for i, (x, y) in enumerate(dataloader):  # loop over dataset
        optimizer.zero_grad()                # reset the gradients
        y_pred = model(x)                    # get output and hidden state
        loss = criterion(y_pred.permute(0,1,2), y)  # compute the loss (change shape as crossentropy takes input as batch_size, number of classes, d1, d2, ...)
        train_loss += loss
        loss.backward()                      # backpropagate
        optimizer.step()                     # update weights 
    return train_loss/len(dataloader)

def valid_lstm_gen(model, criterion, dataloader):
    model.eval()
    valid_loss = 0 
    for i, (x, y) in enumerate(dataloader):
      y_pred = model(x)
      loss = criterion(y_pred.permute(0,1,2), y)
      valid_loss += loss
    return valid_loss/len(dataloader)

def predict_lstm_gen(data, model,input_size, output_size):
    data = data.values
    model.eval()  # set model to evaluation mode
    for i in range(output_size):  # based on the new generated new data
        x = torch.tensor([[[value for value in data[-input_size:]]]], dtype=torch.float32)  # take from dataset and send to device
        y_pred = model(x)  # compute output and hidden state
        last_predict = y_pred # take last output
        # print(last_predict[0,0,0])
        data = np.append(data,last_predict[0,0,0].detach()) # get word corresponding to dataset
    return data


def pre_train(data,base,input_size,target_len):
    price = data
    n_hidden = 15                         
    n_layers = 2
    batch_size = 50

    lr = 2e-4
    n_epochs = 80

    lstm_gen = LSTM_GEN(input_size,n_hidden,n_layers,output_size=target_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_gen.parameters(), lr=lr)

    train_dataset = StockDataset(price[:3000],input_size,base,target_len)
    valid_dataset = StockDataset(price[3001:],input_size,base,target_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  

    # Keep track of losses for plotting
    # liveloss = PlotLosses()
    for epoch in range(n_epochs):
        train_loss = train_lstm_gen(lstm_gen, optimizer, criterion, train_dataloader)
        valid_loss = valid_lstm_gen(lstm_gen, criterion, valid_dataloader)
        logs = {}
        logs['' + 'log loss'] = train_loss.item() # train loss
        logs['val_' + 'log loss'] = valid_loss.item() # val loss
        # liveloss.update(logs)
        # liveloss.draw()
        print(logs)
    return lstm_gen

