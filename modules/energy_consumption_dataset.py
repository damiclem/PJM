# Dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Define a DataLoader for automatically load dataset
class EnergyConsumptionDataset:
    
    # Constructor
    def __init__(self, dataset, col_max=None, col_min=None, lag=5):
        # Call parent constructor
        super().__init__()
        # Define max and min for each column
        self.max = dataset.max(axis=0) if col_max is None else col_max
        self.min = dataset.min(axis=0) if col_min is None else col_min
        # Save normalized dataset reference
        self.dataset = (dataset - self.min) / (self.max - self.min)
        # Save user defined lag
        self.lag = lag
        
    # Define length friend function
    def __len__(self):
        # Return the last row available, given the current lag
        return self.dataset.shape[0] - self.lag
        
    # Define itemgetter
    def __getitem__(self, i):
        # Select a slice of the dataset from row i to row i+lag
        sliced = self.dataset[i:i + self.lag + 1]
        # Turn sliced dataframe into PyTorch Tensor
        sliced = torch.tensor(sliced.values, dtype=torch.float)
        # Return pytorch tensor
        return sliced