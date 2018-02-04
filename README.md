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

```  ```

where the `options` are

- `--processor PROCESSOR` to select a processor  (default SevenPlane)
- `--cores CORES` to use CORES threads (default to the number of processors)
- `--boardsize SIZE` to set the board size of processed games (19 by default)
- `--buffer-size SIZE` to set the size of the board buffer (default 25000) used to randomize the boards in the sets:
  each sampled board is first put in the buffer; when the buffer is full, one board is sampled from the buffer and output.

To control the number of sampled boards, you can use either `--train`, `--validation`
or `--test` with the following argument `GAME_RATIO[:MAX_GAMES[:BOARD_RATIO[:MAX_BOARDS]]]` where

- `GAME_RATIO` is the ratio of sampled games
- `MAX_GAMES` is the maximum number of sampled games
- `GAME_RATIO` is the ratio of sampled boards (within a game)
- `MAX_GAMES` is the maximum number of sampled boards

Each argument is optional; for instance `--train .5::.3` will sample 50% of the games, and among those games, 30% of the boards

## Training

### Direct training

```./ggo direct-policy-train [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] MODEL MODEL_FILE [MODEL OPTIONS]```

where

- `--batchsize BATCHSIZE` gives the training batchsize
- `--checkpoint CHECKPOINT` gives the number of iterations before 
- `--iterations ITERATIONS` is the number of iterations
- `--reset` resets the model instead of continuing to train with it
- `MODEL_PATH` is the file or directory that will contain all the information necessary to run the model
- `MODEL OPTIONS` are options specific to the model (must be serialized with the information)


## Demo

Launch a go game with a demo bot (you have to open a web page)

``` ```

where `MODEL` is a model name (e.g. `gammago.models.idiot`) and `PARAMETERS` 
are the model parameters
