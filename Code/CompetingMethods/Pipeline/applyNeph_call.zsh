#!/bin/zsh

# Cell Line Parameters
cellList=( "H1hesc" )

# Cell Loop
for cell in $cellList
do

    # Parameters
    fosThresh="0.95"
    coordFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/InputRegions/hd.bed"
    dnaseFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/DNase.bw"
    outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/Neph.bed"

    # Execution
    bsub -J $cell"_ANE" -o $cell"_ANE_out.txt" -e $cell"_ANE_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyNeph_pipeline.zsh $fosThresh $coordFileName $dnaseFileName $outputFileName

done


