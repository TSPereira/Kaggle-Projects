import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchsample.transforms as tstf

# Define Net class
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding = 2)   # 28x28
		self.conv1_bn = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding = 2)
		self.conv2_bn = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,64,3,padding=1)
		self.conv3_bn = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64,64,3,padding=1)
		self.conv4_bn = nn.BatchNorm2d(64)
		
		
		self.nr_flat_features = 64 * 7 * 7
		self.fc1 = nn.Linear(self.nr_flat_features, 1024)
		self.fc1_bn = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc2_bn = nn.BatchNorm1d(512)
		self.fc3 = nn.Linear(512, 256)
		self.fc3_bn = nn.BatchNorm1d(256)
		self.fc4 = nn.Linear(256, 10)
	
	def forward(self, x):
		out = F.relu(self.conv1_bn(self.conv1(x)))
		out = F.dropout2d(F.max_pool2d(F.relu(self.conv2_bn(self.conv2(out))), 2), p=0.2)
		out = F.relu(self.conv3_bn(self.conv3(out)))
		out = F.dropout2d(F.max_pool2d(F.relu(self.conv4_bn(self.conv4(out))), 2), p=0.2)
		
		out = out.view(-1, self.nr_flat_features)
		out = F.relu(self.fc1_bn(self.fc1(out)))
		out = F.relu(self.fc2_bn(self.fc2(out)))
		out = F.relu(self.fc3_bn(self.fc3(out)))
		out = F.softmax(self.fc4(out), dim=0)
		
		return out


# A class to prepare our data for the DataLoader can be defined
'''Any custom dataset class has to inherit from the PyTorch dataset Class.
Also, should have __len__ and __getitem__ atributes
set. __init__ method allows to manipulate and transform our raw data'''


class prepData(Dataset):
	def __init__(self, X, Y):
		X = X.reshape((-1, 1, 28, 28))  # Add one channel to use convolution. first dimensions refers to number of images
		self.X = torch.tensor(X, dtype=torch.float32)
		self.Y = torch.tensor(Y)
	
	def __len__(self):
		# Length of our data
		return len(self.Y)
	
	def __getitem__(self, idx):
		# Allows to get a sample from our dataset
		X = self.X[idx]
		Y = self.Y[idx]
		
		sample = {'X': X, 'Y': Y}
		return sample
