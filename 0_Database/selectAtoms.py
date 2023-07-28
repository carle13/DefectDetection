import numpy as np
import glob
import os
import os.path
import sys
import re

import ovito
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *

list_index = sorted(glob.glob('*.indices'))
fIndex = 'disloBStack.indices'
outDir = 'Stack'
inDir = 'DisloB'

indices = np.loadtxt(fIndex, dtype=int)
conditions = ['ParticleIdentifier == '+str(indices[i]) for i in range(len(indices))]
stringCond = ' || '.join(conditions)

os.makedirs('QSelectedNonAverage/'+outDir, exist_ok=True)
list_file = sorted(glob.glob('QValuesNonAverage/'+inDir+'/*.xsf', recursive=True))
for f in list_file:
    pipeline = import_file(f, multiple_frames=True)
    pipeline.modifiers.append(ExpressionSelectionModifier(expression=stringCond))
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())
    data = pipeline.compute()

    export_file(data, f.replace('QValuesNonAverage', 'QSelectedNonAverage').replace(inDir, outDir), "lammps/dump", columns = ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", 'q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'q_7', 'q_8', 'q_9', 'q_10', 'q_11', 'q_12', 'q_13', 'q_14', 'q_15', 'q_16', 'q_17', 'q_18', 'q_19', 'q_20', 'q_21'])