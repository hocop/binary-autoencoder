import os

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

print('Will save data to', hparams['data_path'])

# Create folder if necessary
if not os.path.exists(hparams['data_path']):
    os.makedirs(hparams['data_path'])

# Download and unzip embeddings
embeddings = 'glove.840B.300d'
os.system('wget http://nlp.stanford.edu/data/%s.zip' % embeddings)
os.system('unzip %s.zip' % embeddings)
os.system('rm %s.zip' % embeddings)
# Move them to data path
os.system('mv %s.txt %s' % (embeddings, os.path.join(hparams['data_path'], 'embeddings.txt')))
