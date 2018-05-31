import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchsample.transforms as tstf

# Define Net class
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding = 1)   # 28x28
		self.conv1_bn = nn.BatchNorm2d(3)
		self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding = 1)
		self.conv2_bn = nn.BatchNorm2d(6)
		#self.conv3 = nn.Conv2d(32,64,3,padding=1)
		#self.conv3_bn = nn.BatchNorm2d(64)
		#self.conv4 = nn.Conv2d(64,64,3,padding=1)
		#self.conv4_bn = nn.BatchNorm2d(64)
		
		
		self.nr_flat_features = 6 * 14 * 14
		self.fc1 = nn.Linear(self.nr_flat_features, 120)
		self.fc1_bn = nn.BatchNorm1d(120)
		self.fc2 = nn.Linear(120, 10)
		#self.fc2_bn = nn.BatchNorm1d(512)
		#self.fc3 = nn.Linear(512, 256)
		#self.fc3_bn = nn.BatchNorm1d(256)
		#self.fc4 = nn.Linear(256, 10)
	
	def forward(self, *x):
		x=torch.stack(x)
		out = F.relu(self.conv1_bn(self.conv1(x)))
		out = F.dropout2d(F.max_pool2d(F.relu(self.conv2_bn(self.conv2(out))), 2), p=0.2)
		
		out = out.view(-1, self.nr_flat_features)
		out = F.relu(self.fc1_bn(self.fc1(out)))
		out = F.softmax(self.fc2(out), dim=0)
		
		return out


# A class to prepare our data for the DataLoader can be defined
'''Any custom dataset class has to inherit from the PyTorch dataset Class.
Also, should have __len__ and __getitem__ atributes
set. __init__ method allows to manipulate and transform our raw data'''




class prepData(Dataset):
	def __init__(self, X, Y, input_transform = None):
		self.num_inputs = X.shape[1] if len(X.shape) > 1 else 1     #necessary to use torchsample module trainer
		self.num_targets = Y.shape[1] if len(Y.shape) > 1 else 1    #necessary to use torchsample module trainer
		X = X.reshape((-1, 1, 28, 28))  # Add one channel to use convolution. first dimensions refers to number of images
		self.X = torch.tensor(X, dtype=torch.float32)
		self.Y = torch.tensor(Y)
		self.transform = input_transform
		
	
	def __len__(self):
		# Length of our data
		return len(self.Y)
	
	def __getitem__(self, idx):
		# Allows to get a sample from our dataset
		X = self.X[idx]
		Y = self.Y[idx]
		
		if self.transform:
			X = self.transform(X)
		
		#sample = {'X': X, 'Y': Y}
		sample = [X, Y]
		return sample
