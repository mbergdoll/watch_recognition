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

input_w=32
input_h=52

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear( 3 * input_w * input_h , input_w * input_h )

	def forward(self, x):
		# x => [torch.FloatTensor of size 4x3x520x320]
		x = x.view(-1, 3 * input_w * input_h )
		# x => [torch.FloatTensor of size 4x4992]
		x = self.fc1(x)
		return x