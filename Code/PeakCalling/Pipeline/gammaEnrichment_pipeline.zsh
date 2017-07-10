#!/bin/zsh

# Gamma Enrichment
# Performs gamma enrichment to call peaks in F-seq data.

# Usage:
# ./gammaEnrichment_pipeline.zsh <pValue> <statsFileName> <inputLocation> <outputLocation>

# Parameters:
# <pValue> = p-value to stablish the cutoff.
# <statsFileName> = Location and name of the stats file.
# <inputLocation> = Input location of the genomic signal.
# <outputLocation> = Location of the output and temporary files.

# Imports
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# 0. Initializations
pValue=$1
statsFileName=$2
inputLocation=$3
outputLocation=$4

# 1. Calling peaks
echo "1. Calling peaks"
gammaEnrichment $pValue $statsFileName $inputLocation $outputLocation
echo ""

echo "Script terminated successfully!"
echo "----------------------------------------------"
echo ""





































# PATH
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
# PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
# LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# Input
pValue=$1
mean=$2
var=$3
inputLocation=$4
outputLocation=$5

python gammaEnrichment.py $pValue $mean $var $inputLocation $outputLocation


