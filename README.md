# Autoencoder

This repository is PyTorch implementation of vanilla, sparse and variational autoencoder.

## Introduction

Autoencoder is a type of nueral network consisting of compression and de-compression section. It is primarily used to learn meaningful encodings of given data. 
They have been widely used in the field of image processing, semantic meaning of words, fraud detection and others. Below is a snapshot of the learned representation 
of ```MNIST``` dataset projected on 2D using tSNE:

![only_recon](https://user-images.githubusercontent.com/14811389/109862880-f999cc00-7c86-11eb-8b3f-2134743da23c.png)

## Instruction

The ```autoencoder_model.py``` contains three major classes:

| Class | Autoencoder type|
| ------------- | ------------- |
| auto | Vanilla |
| sparse_auto | Sparse |
| var_auto | Variational |

This implementation has been done on ```MNIST``` dataset at kaggle (https://www.kaggle.com/c/digit-recognizer). 

User can configure data input part in ```read_data.py``` file as per their requirement. All the configurable parameneters are present 
in ```config.py``` file. With proper configuration of ```read.py``` file, execute following command for model training:

```sh
$ python main.py
```

## Requirements

This repository relies on Python 3.5 and PyTorch 1.3.0.

## Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
