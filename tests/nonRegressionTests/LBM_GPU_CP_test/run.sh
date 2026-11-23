#!/bin/bash
TARGET=${1:-TurbineLBM_GPU_CP}
"$TARGET" testCase.prm
python postprocess.py
