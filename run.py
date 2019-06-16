import os
import sys
import time
import datetime

python_v = 'python3'
error_message = 'Valid usage example:\npython3 run.py --actions prepare_dialogs.py train.py smth_else.py --hparams hparams1.json hparams2.json'

try:
    actions = sys.argv[sys.argv.index('--actions') + 1: sys.argv.index('--hparams')]
    hparams = sys.argv[sys.argv.index('--hparams') + 1:]
except:
    actions = [s for s in sys.argv[1:] if '.py' in s]
    hparams = [s for s in sys.argv[1:] if '.json' in s]
    if len(actions) == 0 or len(hparams) == 0:
        print(error_message)
        exit()

commands = []
print('Will run following commands:')
for a in actions:
    for h in hparams:
        commands.append(' '.join([python_v, a, h]))
        print(commands[-1])

timedelta = {}

for cmd in commands:
    start = time.time()
    os.system(cmd)
    timedelta[cmd] = (datetime.timedelta(seconds=int(time.time() - start)))
    print('\nCommand finished:', cmd)
    print('Command took %s' % timedelta[cmd])

print('\nTiming:')
for cmd in commands:
    print(timedelta[cmd], '\t', cmd)
