#!/bin/bash
TARGET=${1:-TurbineLBM_CPU_CP}
mpirun -np 6 "$TARGET" testCase.prm
python postprocess.py
