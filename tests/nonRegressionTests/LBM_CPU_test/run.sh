#!/bin/bash
TARGET=${1:-TurbineLBM_CPU}
mpirun -np 6 "$TARGET" testCase.prm
python postprocess.py
