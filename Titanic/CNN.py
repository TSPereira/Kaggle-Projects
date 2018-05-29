import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PlotFunctions as PF
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from sklearn.model_selection import train_test_split


# Define Net class
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding = 1)   # 22x1
		self.conv2 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, padding = 1)  # Previous 2*2 pooling => 22x1
		self.nr_flat_features = 6 * 22
		self.fc1 = nn.Linear(self.nr_flat_features, 100)
		self.fc2 = nn.Linear(100, 1)
	
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		out = out.view(-1, self.nr_flat_features)
		out = F.sigmoid(self.fc2(F.relu(self.fc1(out))))
		return out


# A class to prepare our data for the DataLoader can be defined
'''Any custom dataset class has to inherit from the PyTorch dataset Class.
Also, should have __len__ and __getitem__ atributes
set. __init__ method allows to manipulate and transform our raw data'''


class prepData(Dataset):
	def __init__(self, X, Y):
		X = X.reshape(
			(-1, 1, 22))  # Add one channel to use convolution. first dimensions refers to number of images
		self.X = torch.from_numpy(X)
		self.Y = torch.from_numpy(Y)
	
	def __len__(self):
		# Length of our data
		return len(self.Y)
	
	def __getitem__(self, idx):
		# Allows to get a sample from our dataset
		X = self.X[idx]
		Y = self.Y[idx]
		
		sample = {'X': X, 'Y': Y}
		return sample
