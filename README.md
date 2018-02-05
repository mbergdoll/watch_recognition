# watch_recognition 

## Objectif

Create an AI that can recognize famous wathces like IWC or Hublot due to a deep neural network and convolutional neural network.

## Prerequisites

The only pre-requisite needed is numpy; you should install your favorite 
machine learning toolkit. At the moment, there is a base class and examples 
for PyTorch.
If you want to donwload image, You have to download MPI.

# Commands
## Downloading images from text file (optional)

``` mpiexec -n 10 python download.py ```

## Pre-processing

Before training, you need to pre-process the data with the command 

``` prepare.py ```

We convert image to jpg format and 32x52 pixels.

## Training

``` train.py ```

## Demo

Analyse an image and return the result

``` demo.py ```

