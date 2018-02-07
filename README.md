# watch_recognition 

## Objectif

Create an AI that can recognize famous watches like IWC or Hublot due to a deep neural network and convolutional neural network.

## Prerequisites

The only pre-requisite needed is numpy; you should install your favorite 
machine learning toolkit. At the moment, there is a base class and examples 
for PyTorch.
If you want to donwload image, You have to download MPI.

# Commands
## Downloading images from text file (optional)

``` mpiexec -n 2 python download.py ```

## Pre-processing

All images will classified n train and val folder and in class.
Before training, you need to pre-process the data with the command:

``` prepare.py ```

We convert image to jpg format and 32x52 pixels.

## Training

``` train.py ```

## Demo: It gives statistical data from the results of the training.

``` demo.py ```

## Analyse an image and return the result: TODO

``` compare.py --image IMAGE ```

where IMAGE is the path of the image. It searches the good brand associated with the watch submitted.
