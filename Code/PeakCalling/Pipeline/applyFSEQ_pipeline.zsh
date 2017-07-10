#!/bin/zsh

# Apply FSEQ
# Applies FSEQ on a genomic signal obtained by ChIP-seq or DNase-seq

# Usage:
# ./applyFSEQ_pipeline.zsh <featLen> <bff> <iff> <treatmentLocation> <outputLocation>

# Parameters:
# <featLen> = Feature length required by F-seq.
# <bff> = Background bff files.
# <iff> = Ploidy iff files.
# <treatmentLocation> = Location of treatment files.
# <outputLocation> = Location of the output and temporary files.

# Imports
export JAVA_TOOL_OPTIONS=-Xmx20000m
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# 0. Initializations
featLen=$1
bff=$2
iff=$3
treatmentLocation=$4
outputLocation=$5

# 1. Creating merged bed input for treatment
echo "1. Creating merged bed input for treatment"
treatmentList=($treatmentLocation*.bam)
for i in {1..$#treatmentList}
do
    fn=$treatmentList[$i]
    mkdir -p $treatmentLocation"FSEQ_"$name"/"
    bamToBed -i $fn > $treatmentLocation"FSEQ_"$name"/pre"$i".bed"
done
bedMerge $outputLocation"treatment.bed" $treatmentLocation"FSEQ_"$name"/pre"*".bed"
rm $treatmentLocation"FSEQ_"$name"/pre"*".bed"
echo ""

# 2. Performing the motif matching
echo "2. Performing the motif matching"
fseq -l $featLen -of "wig" -b $bff -p $iff -d $outputLocation -o $outputLocation -v "treatment.bed"
rm $outputLocation"treatment.bed"
echo ""

echo "Script terminated successfully!"
echo "----------------------------------------------"
echo ""


