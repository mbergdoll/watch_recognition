# watch_recognition 

## Objectif

Create an AI that can recognize famous watches like IWC or Hublot by using a deep neural network and a convolutional neural network.

Principle of the project:

<p align="center">
  <img src="https://raw.githubusercontent.com/mbergdoll/watch_recognition/master/Capture.PNG" width="500px"/>
</p>

###1st step: detect watches in a image: OpenCV

I have used 1,500 images of watches and 10,000 images whithout watches inside.
The result is a xml file (watches.xml)

###2nd step: train a neural network to label watches: Pytorch

I train the CNN with 1,500 watches of 8 different classes.

## Prerequisites

The only prerequisite needed is numpy. You should install your favorite machine learning toolkit. 
At the moment, there are a base class and examples for PyTorch and OpenCV.
If you want to download image from internet, you have to download MPI.

# Commands
## Downloading images from a list of urls (optional)

``` mpiexec -n 2 python download.py ```

## Pre-processing

Before training, you need to pre-process the data with the command:

``` prepare.py ```

We convert image to jpg format and 32x52 pixels. Images are labelled for the training.

## Training

``` train.py --iterations 20 ```

## Analyse an image and return the result:

``` compare.py --image IMAGE ```

where IMAGE is the path of the image. It searches the good brand associated with the watch submitted.

## Limits:

My database contains only 1520 watches. Watches with different positions are not represented in the database. So I have bad results with the watches tested that have "noise". I can ameliorate the result with other images of watches in different points of view.

