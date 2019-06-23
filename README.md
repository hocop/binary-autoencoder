# Binary Autoencoder for Text Modeling.
This repository contains main language modeling experiments from my paper (unpublished yet):  
* binary autoencoder (main subject of paper)  
* variational autoencoder (strong baseline)  
* bottleneck autoencoder (weak baseline)  
* just RNN language model  

## Setup
Install requirements: 

    sudo pip3 install -r requirements.txt
Tensorflow is also required. I use version `tensorflow-gpu==1.9.0`.  
Then download datasets and embeddings:

    # download the data (every experiment uses the same data)
    python3 run.py download_dataset.py download_embeddings.py experiment_configs/binary.json
## How to reproduce experiments
For example, binary autoencoder.  
Open `experiment_configs/binary.json`. Look at hyperparameters. Note that all pathes listed in this file will be created on your computer.

    # train autoencoder
    python3 train.py experiment_configs/binary.json
    # evaluate on test set
    python3 evaluate.py experiment_configs/binary.json
    # encode test set to binary vectors
    python3 encode_file.py experiment_configs/binary.json
    # view the resulting file
    head ~/data/language_modeling/model_output/binary/vectors.txt
    # decode binary vectors back to text
    python3 decode_file.py experiment_configs/binary.json
    # compare the original text and the decoded text
    sdiff \
        ~/data/language_modeling/data/test.txt \
        ~/data/language_modeling/model_output/binary/decoded.txt \
        | head -n 100

Other experiments can be reproduced in the same way. Configs for them are also stored in `experiment_configs`. Training data and embeddings do not need to be downloaded twice.
