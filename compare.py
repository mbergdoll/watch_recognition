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
import torch.nn.functional as F
import sys
import Model_CNN
import PIL
from PIL import Image

from constante import *

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


######################################################################
# Load Model
# ^^^^^^^^^^^^^^^^^^^^^^
def load_model():
	import copy
	import pickle
	with open('parameters.dat', 'rb') as fid:
	    net = pickle.load(fid)

	print("Model & parameters loaded...")
	return net

#######################################################################
# The image to compare in JPG Format with 32x52 pixels
#^^^^^^^^^^^^^^^^^^^^^^

if( len(sys.argv)!=3 ):
	print("./compare.py --image test/1.jpg")
else:
	path = sys.argv[2] # ./compare.py --image Image
	net = Model_CNN.Net()
	net = load_model()

	print("path: "+str(path) )

	image = Image.open( path )

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
	
	#dataiter = iter( dataloaders['val'] )
	#images, labels = dataiter.next() # images => [torch.FloatTensor of size 4x3x52x32]
	#imshow(torchvision.utils.make_grid(images))
	#outputs = net(Variable(images))
	#_, predicted = torch.max(outputs.data, 1)
	#print('Predicted: %s' % class_names[predicted[0]] ) 

	image = Image.open( path )
	to_tensor = transforms.Compose([
	        transforms.Scale((input_w, input_h)),
	        transforms.ToTensor(),
	        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	    ])
	img = to_tensor( image )
	image_initiale = Image.open( path )
	to_tensor = transforms.Compose([
	        transforms.ToTensor(),
	    ])
	img_initale = to_tensor( image_initiale )
	#print( img )
	# print images 
	imshow(torchvision.utils.make_grid(img_initale))
	outputs = net(Variable(img))
	_, predicted = torch.max(outputs.data, 1)
	print('Predicted: %s' % class_names[predicted[0]] ) 