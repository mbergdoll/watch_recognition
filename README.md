# watch_recognition 

## Objectif

Create an AI that can recognize famous watches like IWC or Hublot due to a deep neural network and convolutional neural network.

Principles of the project:

<p align="center">
  <img src="https://raw.githubusercontent.com/mbergdoll/watch_recognition/master/Capture.PNG" width="50%"/>
</p>

###1st step: detect watches in images: OpenCV

I use 1,500 images of watches and 10,000 non-watches images.
The result is an xml file (watches.xml).
Results:
  - Statsistics: TO DO
Limits:
  - Number of images
  - Diversification

###2nd step: train a neural network: Pytorch
I train the CNN with 1,500 watches of 8 different classes.
Results:
  - Statsistics: TO DO
Limits:
  - Number of images

## Prerequisites

The only prerequisite needed is numpy. You should install your favorite machine learning toolkit. 
At the moment, there are a base class and examples for PyTorch.
OpenCV
If you want to download image, You have to download MPI.

# Commands
## Downloading images from text file (optional)

``` mpiexec -n 2 python download.py ```

## Pre-processing

All images will classify n train and val folder and in class.
Before training, you need to pre-process the data with the command:

``` prepare.py ```

We convert image to jpg format and 32x52 pixels.

## Training

``` train.py --iterations 20 ```

## Analyse an image and return the result:

``` compare.py --image IMAGE ```

where IMAGE is the path of the image. It searches the good brand associated with the watch submitted.

## Limits:

My database contains only 1520 watches with the same position by class. Watches with different position are not represented in the database. So I have bad results with the watches tested that have "noise". I can ameliorate the result with other images of watches tilted and in different points of view for example.

