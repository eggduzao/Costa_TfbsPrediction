#!/bin/zsh

###################################################################################################
# A. Operations:
#    1. Create extended footprints with merging
###################################################################################################

# Parameters
il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"
inputList=( $il"H1hesc/Boyle.bed" $il"K562/Boyle.bed" )

# Execution
for inputFile in $inputList
do
    ./fixBoyleResults_pipeline.zsh $inputFile
done


