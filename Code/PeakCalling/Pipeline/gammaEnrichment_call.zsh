#!/bin/zsh

# Parameters
sigLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/FSEQ/K562/All/"

# Variations
pValueList=("0.01" "0.025" "0.05" "0.1")
currDir=$PWD
cd $sigLoc
factorList=(*)
cd $currDir

# Execution
for factor in $factorList
do
    for pV in $pValueList
    do
        bsub -J $factor"_"$pV"_GAMMA" -o $factor"_"$pV"_GAMMA_out.txt" -e $factor"_"$pV"_GAMMA_err.txt" -W 40:00 -M 4096 -S 100 -P izkf ./gammaEnrichment_pipeline.zsh $pV $sigLoc$factor"/stats.txt" $sigLoc$factor"/signal/" $sigLoc$factor"/"
    done
done


