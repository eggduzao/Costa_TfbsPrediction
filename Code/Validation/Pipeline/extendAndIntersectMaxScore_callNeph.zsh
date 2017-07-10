#!/bin/zsh

################################################
### NEPH
################################################

cellList=( "H1hesc" "K562" )

# Cell Loop
for cell in $cellList
do

  # Predictions Parameters
  predList=( "Neph" )

  # Predictions Loop
  for pred in $predList
  do

    # Extension Paramers
    extList=( "5" )

    # Extension Loop
    for ext in $extList
    do
            
      # Parameters
      mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"$cell"/TagCount_DU.bed"
      predFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$pred".bed"
      outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"$cell"/"$pred"_DU_"$ext".bed"

      # Execution
      bsub -J $cell"_"$pred"_EIMN" -o $cell"_"$pred"_EIMN_out.txt" -e $cell"_"$pred"_EIMN_err.txt" -W 5:00 -M 12000 -S 100 -R "select[hpcwork]" ./extendAndIntersectMaxScore_pipeline.zsh $ext $ext $mpbsFileName $predFileName $outputFileName
      # -P izkf

    done
  done
done


