#!/usr/bin/env python

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
data_dir = 'watches_classed'
input_w=32
input_h=52
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
	    transforms.Scale((input_w, input_h)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
	    transforms.Scale((input_w, input_h)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import Model_CNN

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

net = Model_CNN.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize
print('Training...')
abscisse=[]
res=[]
c=0
trainloader = dataloaders['train']
for epoch in range(1):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, labels = data
		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.data[0]
		if i % 10 == 9:    # print every 2000 mini-batches
			print('[iteration: %d] loss: %.3f' % (c+1, running_loss / 10) )
			c=c+1
			abscisse = abscisse + [ c ]
			res = res + [ running_loss / 10 ]
			running_loss = 0.0

print('Finished Training')

p1=plt.plot( abscisse , res ,color='blue')
plt.title("linear model")
plt.legend(["train"])
plt.show()

# save Model
import copy
import pickle

# copy you entirely object and save it 
saved_trainer = copy.deepcopy( net )
with open(r"parameters.dat", "wb") as output_file:
    pickle.dump(saved_trainer, output_file)
