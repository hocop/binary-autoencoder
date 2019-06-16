import nltk
import numpy as np
import os
import string

nltk.download('punkt')

# Converts lines of text to sequences of int
class Encoder:
    def __init__(self, hparams):
        # load vocabulary from glove embeddings file
        self.vocab = ['<pad>', '<EOS>', '<unk>']
        for i, l in enumerate(open(os.path.join(hparams['data_path'], 'embeddings.txt'))):
            if i > hparams['vocab_size']:
                break
            self.vocab.append(l.split()[0])
        # create reversed vocabulary
        self.word2idx = {self.vocab[i]: i for i in range(len(self.vocab))}
    
    def encode(self, line):
        # convert string to sequence of integers
        return [self.word2idx.get(word.strip(), self.word2idx['<unk>']) for word in nltk.word_tokenize(line)]
    
    def decode(self, numbers):
        # convert sequence of integers back to string
        res = ''
        for i, word in enumerate(self.decode_list(numbers)):
            if i != 0 and not word in string.punctuation:
                res += ' '
            res += word
        return res
    
    def decode_list(self, numbers):
        lst = []
        for n in numbers:
            if n == 1:
                break
            lst.append(self.vocab[n])
        return lst

def gen_batch(buf, hparams, mode, keep_order):
    id0 = min(list(buf))
    batch = [buf[id0]] # start with the oldest line in buffer
    idxes = [id0]
    max_len = len(buf[id0])
    if keep_order:
        # process lines in the same order as in input file (for predictions)
        order = sorted(list(buf))
    else:
        # generate batch in memory-efficient way (for training)
        order = sorted(list(buf), key=lambda x: abs(len(buf[x]) - len(buf[id0])))
    # build batch
    for idx in order:
        if idx == id0:
            continue
        max_len = max(max_len, len(buf[idx]))
        if (len(batch) + 1) * max_len > hparams['tokens_per_batch']:
            break
        batch.append(buf[idx])
        idxes.append(idx)
    # add zeros
    batch_out = pad_sequences([seq + [1] for seq in batch]) # 1 means <EOS> - end of sequence
    batch = pad_sequences(batch, batch_out.shape[1])
    # delete used lines from buffer
    for idx in idxes:
        del buf[idx]
    return batch, batch_out

def batch_generator(hparams, mode, start_from=0, keep_order=False):
    assert mode in ['train', 'test', 'dev']
    encoder = Encoder(hparams)
    filename = os.path.join(hparams['data_path'], mode + '.txt')
    len_file = 0
    for _ in open(filename):
        len_file += 1
    buf = {}
    # Read input file
    for i, line in enumerate(open(filename)):
        if i < start_from:
            continue
        line = encoder.encode(line.strip())
        buf[i] = line
        # make batch of data
        if len(buf) >= hparams['buffer_size']:
            yield gen_batch(buf, hparams, mode, keep_order)
    while len(buf) != 0:
        yield gen_batch(buf, hparams, mode, keep_order)

def pad_sequences(arr, ml=None):
    max_len = ml or max([len(seq) for seq in arr])
    # pad
    for seq in arr:
        for _ in range(max_len - len(seq)):
            seq.append(0)
    return np.array(arr)























