#!/bin/bash 
TARGET=${1:-TurbineLBM_CPU}

# Generate the case using the python helper
cd CaseGeneration
python generate.py

# Now run the case in a dedicated folder
mv NeutralABL_Andren_Files ../
cd ../NeutralABL_Andren_Files/
mpirun -np 6 "$TARGET" NeutralABL_Andren_Files_30cells.prm

# Postprocess the case in an existing dedicated folder
cd ../PostProcessing/
python postprocessResults.py


