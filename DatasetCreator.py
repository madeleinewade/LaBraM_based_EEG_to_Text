from math import nan
import os
import mne
import numpy as np
import pandas as pd
import pickle as pkl
import random
from huggingface_hub import snapshot_download

folder = '/scratch/alpine/mawa5935/DL/.cache/datasets--madeleinewade--Murphy2022Data/snapshots/4783e66678539b46776b9157a20e41aed009c60a/'

empty = np.zeros(276)

x = []
y = []
OOV = []
vocabulary = {}
word_idx = 0
full_sentences = []
full_sentences_eeg = []

POS_map = {'PAD': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4,
'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9,
'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14,
'VERB': 15, 'X': 16}

print('Creating dataset...')
for f in os.listdir(folder):
    if '.fif' not in f:
        continue
    print('Loading file: ', f)
    #load epochs
    filename = folder+f
    epochs = mne.read_epochs(filename)
    epochs_df = epochs.to_data_frame()
    metadata = epochs.metadata
    metadata['pos'] = metadata['pos'].astype('category')
    metadata['pos'] = metadata['pos'].map(POS_map)
    metadata['len'] = metadata['len'].astype('int')
    i = 0
    for w in metadata['len']:
        if w>20:
            metadata.loc[i, 'len'] = 20
        i = i + 1
    metadata['freq'] = metadata['freq']*100
    for w in metadata['freq']:
        if w>300:
            metadata.loc[i, 'freq'] = 300
        i = i + 1
    metadata['freq'] = metadata['freq'].fillna(0)
    metadata['freq'] = metadata['freq'].astype('int')

    metadata['sent_ident'] = metadata['sent_ident'].astype(str)

    for s in np.unique(metadata['sent_ident']):
        sentence_meta = metadata[metadata['sent_ident']==s]
        sentence_eeg = epochs_df[epochs_df['epoch'].isin(sentence_meta.index)]
        sentence_x = np.zeros((32,59,200))
        sentence_y = np.zeros((32, 4))
        length = len(np.unique(sentence_eeg['epoch']))

        if length<5 or length>29:
            #skip sentences that are too long or too short
            continue

        i = 0

        full_sent = ''
        for w in np.unique(sentence_eeg['epoch']):

            meta = sentence_meta.loc[w]

            word_vec = np.empty(4)

            full_sent = full_sent + ' ' + meta['word']

            if meta['word'] not in vocabulary:
                vocabulary[meta['word']] = word_idx
                word = word_idx
                word_idx = word_idx + 1
            else:
                word = vocabulary[meta['word']]

            word_vec[0] = meta['pos']
            word_vec[1] = meta['len']
            word_vec[2] = meta['freq']
            word_vec[3] = word


            y.append(word_vec)
            eeg = sentence_eeg[sentence_eeg['epoch'] == w]
            eeg = eeg.drop(columns=['epoch', 'condition', 'time'])
            eeg = eeg.to_numpy()
            eeg = eeg.T
            eeg = eeg[:, 76:276]

            x.append(eeg)

            i = i + 1

        full_sentences.append(full_sent)


print('Creating x file...')
f= open('x.pkl', 'wb')
pkl.dump(x, f)
f.close()

print('Creating y file...')
f= open('y.pkl', 'wb')
pkl.dump(y, f)
f.close()

print('Creating vocabulary file...')
f= open('vocab.pkl', 'wb')
pkl.dump(vocabulary, f)
f.close()


print('Creating full sentences file...')
f= open('full_sentences.pkl', 'wb') 
pkl.dump(full_sentences, f)
f.close()

print('Creating train/val/test split...')
n = len(x)
idx = list(range(n))
random.shuffle(idx)

train_idx = idx[0:int(np.round(n*0.8))]
val_idx = idx[int(np.round(n*0.8)):int(np.round(n*0.9))]
test_idx = idx[int(np.round(n*0.9)):len(idx)]

x_train = []
x_val = []
x_test = []

y_train = []
y_val = []
y_test = []

for i in range(n):
    if i in train_idx:
        x_train.append(x[i])
        y_train.append(y[i])
    elif i in test_idx:
        x_test.append(x[i])
        y_test.append(y[i])
    elif i in val_idx:
        x_val.append(x[i])
        y_val.append(y[i])
    else:
        print('error at ', i)
        print(x[i].shape)


print('Saving train/val/test split...')
f= open('x_train.pkl', 'wb')
pkl.dump(x_train, f)
f.close()

f= open('y_train.pkl', 'wb')
pkl.dump(y_train, f)
f.close()

f= open('x_val.pkl', 'wb')
pkl.dump(x_val, f)
f.close()

f= open('y_val.pkl', 'wb')
pkl.dump(y_val, f)
f.close()

f= open('x_test.pkl', 'wb')
pkl.dump(x_test, f)
f.close()

f= open('y_test.pkl', 'wb')
pkl.dump(y_test, f)
f.close()
