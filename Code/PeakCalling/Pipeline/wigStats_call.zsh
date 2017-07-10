#!/bin/zsh

# Parameters
sigLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/FSEQ/K562/All/"

# Variations
currDir=$PWD
cd $sigLoc
factorList=(*)
cd $currDir

# Execution
for factor in $factorList
do
    bsub -J $factor"_STATS" -o $factor"_STATS_out.txt" -e $factor"_STATS_err.txt" -W 40:00 -M 4096 -S 100 -P izkf ./wigStats_pipeline.zsh $sigLoc$factor"/signal/" $sigLoc$factor"/"
done


