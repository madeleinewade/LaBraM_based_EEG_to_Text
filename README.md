# LaBraM based Modular EEG to text decoder model
Work in progress!

Finetuning adapted from orginal LaBraM model to be used on dataset from Murphey et al. 2022. 

Model uses a modular architechture that decodes a given word's part of speech, length, and frequency (each a finetuned instance of LaBraM) before decoding the entire sentence through a transformer. 
