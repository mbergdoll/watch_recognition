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
import time
import os
import copy

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int,default=10,help="number of iterations")
args = parser.parse_args()
max_it = args.iterations
print("iterations:"+str(max_it) )

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
from constante import *

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

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print("number of classes:"+str(number_of_classes) )
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
print("size:"+str(3)+"x"+str(input_w)+"x"+str(input_h) )

abscisse=[]
res=[]
res2=[]
c=0
trainloader = dataloaders['train']
valloader = dataloaders['val']
n=0
max_train=0
for i, data in enumerate(trainloader, 0):
	max_train = max_train+1
max_val=0
for i, data in enumerate(valloader, 0):
	max_val = max_val+1

print( "n° of training:"+str(max_train) )
print( "n° of validation:"+str(max_val) )
print('Training...')

for it in range(max_it):
	taux=0
	sum_labels = 0
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
		#print( outputs.data.numpy()[0] )
		#print( "*"+str( outputs.data.numpy().argmax() ) )
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		#print(loss)
		running_loss += loss.data[0]
		#print(str(i+1)+"/"+str(max_train))
		#print( " labels :"+str(labels.data.numpy()[0]) )
		#print( " labels~:"+str(outputs.data.numpy()[0].argmax()) )
		if( labels.data.numpy()[0] != outputs.data.numpy()[0].argmax() ):
			sum_labels+=1
		else:
			sum_labels+=0
	taux = ( sum_labels / max_train )
	
	print("--- Itération %d ---" % it)
	print("Apprentissage: loss: %.7f (error rate=%.1f%%)" % (running_loss/max_train, 100*taux ) ) 
	res = res + [ taux ]

	taux=0
	sum_labels = 0
	running_loss2 = 0.0
	for i, data in enumerate(valloader, 0):
		# get the inputs
		inputs, labels = data
		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)
		# zero the parameter gradients
		#optimizer.zero_grad()
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		#optimizer.step()
		# print statistics
		running_loss2 += loss.data[0]
		#print(str(i+1)+"/"+str(max_val))
		if( labels.data.numpy()[0] != outputs.data.numpy()[0].argmax() ):
			sum_labels+=1
		else:
			sum_labels+=0
	taux = ( sum_labels / max_val )
	print("Validation:    loss: %.7f (error rate=%.1f%%)\n" % (running_loss2/max_val, 100*taux ) ) 
	res2 = res2 + [ taux ]
	c=c+1
	abscisse = abscisse + [ c ]

print('Finished Training...')

import matplotlib.pyplot as plt
p1=plt.plot( abscisse , res ,color='blue')
p1=plt.plot( abscisse , res2 ,color='red')
plt.title("linear model")
plt.legend(["train","val"])
plt.show()

# save Model
import copy
import pickle

# copy you entirely object and save it 
saved_trainer = copy.deepcopy( net )
with open(r"parameters.dat", "wb") as output_file:
    pickle.dump(saved_trainer, output_file)
