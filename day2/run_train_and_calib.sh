#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PYTHON=$(which python3 || which python)
echo "Using Python: $PYTHON"
$PYTHON day2/train_v2.py --model efficientnet_b0 --img 256 --epochs 12 2>&1 | tee day2/train_log.txt
$PYTHON day2/calibrate.py 2>&1 | tee day2/calib_log.txt

