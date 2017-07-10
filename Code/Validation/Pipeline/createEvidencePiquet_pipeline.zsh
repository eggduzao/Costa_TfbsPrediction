#!/bin/zsh

# Create Evidence Piquet
# Creates Piquet-based positive and negative MPBS datasets based on ChIP-seq TFBS.

# Usage:
# ./createEvidencePiquet_pipeline.csh <peakExt> <negExt> <mpbsName> <tfbsSummitFileName> <treatFileName> <controlFileName> <mpbsFileName> <outputLocation>

# Parameters:
# <peakExt> = Extension of the peaks summit (total window length).
# <negExt> = Extension to count signals to create negative set (total).
# <mpbsName> = Name of the MPBS.
# <tfbsSummitFileName> = Location + name of the TFBS ChIP-seq summit file.
# <treatFileName> = Location + name of the ChIP-seq treatment signal.
# <controlFileName> = Location + name of the ChIP-seq control signal.
# <mpbsFileName> = File containing MBPSs.
# <outputLocation> = Location of the output and temporary files.

# Imports
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# 0. Initializations
peakExt=$1
negExt=$2
mpbsName=$3
tfbsSummitFileName=$4
treatFileName=$5
controlFileName=$6
mpbsFileName=$7
outputLocation=$8

# 1. Creating evidence set
echo "1. Creating evidence set"
createEvidencePiquet $peakExt $negExt $mpbsName $tfbsSummitFileName $treatFileName $controlFileName $mpbsFileName $outputLocation
echo ""

echo "Script terminated successfully!"
echo "----------------------------------------------"
echo ""


