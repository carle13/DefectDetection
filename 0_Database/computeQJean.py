import os
import glob

directory = 'Input/DisloA/*.xsf'
indices = 'DisloA.indices'
output = 'QValues'

executable = '/home/umet/Software/AtomHIC/build/bin/GetSteinhardt'

os.makedirs(directory.replace('Input', output), exist_ok=True)

list_file = sorted(glob.glob(directory))
for f in list_file:
    if os.path.exists(f.replace('Input', output)):
        print(f.replace('Input', output), '   Skipped')
        continue
    os.system(executable+' '+f+' 1 '+indices+' DisloA Forsterite '+f.replace('Input', output)+' Multi Multi')