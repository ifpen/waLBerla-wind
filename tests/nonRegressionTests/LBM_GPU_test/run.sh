#!/bin/bash
TARGET=${1:-TurbineLBM_GPU}
"$TARGET" testCase.prm
python postprocess.py
