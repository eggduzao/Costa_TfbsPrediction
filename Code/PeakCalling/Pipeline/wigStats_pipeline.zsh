#!/bin/zsh

# Wig Stats
# Evaluates statistics on wig files.

# Usage:
# ./wigStats_pipeline.zsh <inputLocation> <outputLocation>

# Parameters:
# <inputLocation> = The location to the wig files.
# <outputLocation> = Location of the output and temporary files.

# Imports
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# 0. Initializations
inputLocation=$1
outputLocation=$2

# 1. Evaluating statistics
echo "1. Evaluating statistics"
wigStats $inputLocation $outputLocation
echo ""

echo "Script terminated successfully!"
echo "----------------------------------------------"
echo ""


