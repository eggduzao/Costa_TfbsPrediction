#!/bin/zsh

###################################################################################################
# A. Operations:
#    1. Extend and merge Nephs footprints
###################################################################################################

# Parameters
il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"

inputList=( $il"H1hesc/Neph.bed" )
outputList=( $il"H1hesc/Extension/" )

# Execution
for i in {1..$#inputList}
do
    #bsub -J "FCR" -o "FCR_out.txt" -e "FCR_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" 
    ./fixNephResults_pipeline.zsh $inputList[$i] $outputList[$i]
done


