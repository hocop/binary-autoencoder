import sys
import json
import getpass
import os
import filecmp

def replace_names(where, name, target):
    if type(where) == str and name in where:
        return where.replace(name, target)
    if type(where) == dict:
        for val in where:
            where[val] = replace_names(where[val], name, target)
        return where
    if type(where) == list:
        for val_idx in range(len(where)):
            where[val_idx] = replace_names(where[val_idx], name, target)
        return where
    return where

def loader_func(hparams_path, is_parent=False):
    try:
        hparams = json.load(open(hparams_path))
    except:
        print('Cannot load', hparams_path)
        raise BaseException('cannot load hparams')
    # maybe derive from some other hparams file
    if 'derive_from' in hparams:
        parent = loader_func(hparams['derive_from'], is_parent=True)
        for fn in hparams:
            parent[fn] = hparams[fn]
        hparams = parent
    # username
    if (not 'user_name' in hparams) or hparams['user_name'] == 'default':
        hparams['user_name'] = getpass.getuser()
    # replace all <smth> substrings to hparams["smth"]
    if not is_parent:
        for p1 in hparams:
            hparams = replace_names(hparams, '<' + p1 + '>', str(hparams[p1]))
    return hparams

try:
    hparams = loader_func(sys.argv[1])
except BaseException as e:
    #print('\nERROR: valid hparams file must be specified as first argument\n')
    #hparams = loader_func(sys.argv[1])
    hparams = 'not specified'

def PrintHparamsInfo(hparams):
    def bool_to_str(val):
        if val:
            return 'true'
        else:
            return 'false'
    ESCAPE_INFO = '\033[38;5;209m'
    ESCAPE_TITLE = '\033[38;5;123m'
    ESCAPE_DATA = '\033[38;5;72m'
    ESCAPE_FILE = '\033[38;5;118m'
    ESCAPE_OFF = '\033[0m'
    import __main__
    print(ESCAPE_TITLE + 'Running ' + ESCAPE_FILE +  __main__.__file__ + ESCAPE_TITLE + '; model: ' + ESCAPE_INFO + hparams['model_name'])
    print(ESCAPE_OFF)




