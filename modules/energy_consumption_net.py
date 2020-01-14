# Dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Define new neural network lstm class for time series forecasting
class EnergyConsumptionNet(nn.Module):
    
    # Constructor
    def __init__(self, input_size, hidden_size, num_layers, dropout=0, rnn=nn.LSTM):
        # Call parent constructor
        super().__init__()
        # Define recurrent layer
        self.rnn = rnn(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True
        )
        # Define output layer
        self.out = nn.Linear(hidden_size, 1)
        
    # Forward function
    def forward(self, x, state=None):
        # Run input through recurrent layer
        x, state = self.rnn(x, state)
        # Run input through recurrent layer and return result
        return self.out(x), state
    

# Define function for training a batch
def train_batch(net, batch, optimizer, loss_fn=nn.MSELoss()):
    # Clear previous gradient
    net.zero_grad()
    optimizer.zero_grad()
    # Define inputs and targets
    net_in = batch[:, :-1, :]  # Input
    true_out = batch[:, -1, 0]  # Target
    # Compute output
    net_out, _ = net(net_in)
    # Compute loss
    loss = loss_fn(net_out[:, -1, 0], true_out)  # Compute loss
    loss.backward()  # Make backward update
    optimizer.step()  # Make optimizer update
    # Return computed loss
    return float(loss.data)


# Define a function for training epochs
def train_epochs(net, dataset, batch_size, num_epochs, optimizer, loss_fn=nn.MSELoss(), verbose=False):
    # Create a DataLoader object out of dataset
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # Define a container for training and test loss
    train_loss = list()
    # Get device where the net is stored
    device = next(net.parameters()).device
    # Loop through every epoch
    for epoch in range(num_epochs):
        # Set net in training mode
        net.train()
        # Initialize train loss
        train_loss.append([])
        # Loop through every batch in DataLoader object (iterator)
        for i, batch_ in enumerate(dataloader):
            # Move current batch to selected device
            batch = batch_.to(device)
            # Compute loss for current batch
            loss = train_batch(net, batch, optimizer, loss_fn=loss_fn)
            # Update train loss for current batch
            train_loss[-1].append(loss)
        # After every epoch, output mean loss
        if verbose: 
            print('Epoch nr {:04d} with mean loss: {:.5f}'.format(epoch + 1, np.mean(train_loss[-1])))
    # Return train and test losses
    return train_loss