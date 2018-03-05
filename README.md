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

``` train.py --iterations 20 ```

## Analyse an image and return the result:

``` compare.py --image IMAGE ```

where IMAGE is the path of the image. It searches the good brand associated with the watch submitted.

## Limits:

My database contains only 1520 watches with the same position by class. Watches with different position are not represented in the database. So I have bad results with the watches tested that have "noise". I can ameliorate the result with other images of watches inclinated and in different points of view for example.

