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

import Model_CNN

data_dir = 'watches_classed'
input_w=32
input_h=52


######################################################################
# Load Model
# ^^^^^^^^^^^^^^^^^^^^^^
import copy
import pickle
with open('parameters.dat', 'rb') as fid:
    net = pickle.load(fid)

print("Model & parameters loaded...")


#######################################################################
# The image to compare in JPG Format with 32x52 pixels
#^^^^^^^^^^^^^^^^^^^^^^

if( len(sys.argv)!=3 ):
	print("./compare.py --image Image")
else:
	path = sys.argv[3] # ./compare.py --image Image