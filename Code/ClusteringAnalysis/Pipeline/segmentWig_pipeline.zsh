#!/bin/zsh

# Segment Wig
# Performs a wig summary

# Usage:
# ./segmentWig_pipeline.zsh <anType> <windowSize> <normFactorList> <helpFileName> <wigList> <outputLocation>

# Parameters:
# <anType> = Can be 'all' or 'bed'.
# <windowSize> = If 'all' then it is the window size to read the genome.
# <normFactorList> = List of normalizing factors separated by comma.
# <helpFileName> = If 'all' then chrom.sizes. If 'bed' then a coord. file.
# <wigList> =  List of wig files.
# <outputLocation> =  Location of the output and temporary files.

# Imports
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/:/work/eg474423/ig440396_dendriticcells/exp/motifanalysis/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# 0. Initializations
anType=$1
windowSize=$2
normFactorList=$3
helpFileName=$4
wigList=$5
outputLocation=$6

# 1. Creating summary
echo "1. Creating summary"
segmentWig $anType $windowSize $normFactorList $helpFileName $wigList $outputLocation
echo ""

echo "Script terminated successfully!"
echo "----------------------------------------------"
echo ""


