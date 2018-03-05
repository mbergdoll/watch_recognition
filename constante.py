#!/usr/bin/env python
# encoding: utf-8
import os

data_dir = 'watches_classed'
input_w=32
input_h=52
list_classes = os.listdir("./watches_classed/train")
number_of_classes = len( list_classes )