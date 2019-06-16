import os
import requests

from load_hparams import hparams, PrintHparamsInfo
PrintHparamsInfo(hparams)

print('Will save data to', hparams['data_path'])

# Create folder if necessary
if not os.path.exists(hparams['data_path']):
    os.makedirs(hparams['data_path'])

# Download and unzip data
href = requests.get(hparams['dataset_link']).json()['href']
os.system('wget "%s" -O dataset.zip' % href)
os.system('mkdir tmp')
os.system('unzip dataset.zip -d tmp')
os.system('rm dataset.zip')
os.system('mv tmp/* %s' % hparams['data_path'])
os.system('rm -r tmp')
