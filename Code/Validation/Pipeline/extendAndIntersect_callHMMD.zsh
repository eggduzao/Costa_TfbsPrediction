#!/bin/zsh

################################################
### DNASE+HISTONES HMM PREDICTIONS
################################################

# Lab Parameters
labList=( "DU" "UW" )

# Lab Loop
for lab in $labList
do

  # Cell Parameters
  if [[ $lab == "DU" ]]; then
    cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
  elif [[ $lab == "UW" ]]; then
    cellList=( "HepG2" "Huvec" "K562" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

      # Predictions Parameters
      if [[ $lab == "DU" ]]; then
        predList=( "HMM_DU_BIAS_D13_Model"$cell )
      elif [[ $lab == "UW" ]]; then
        predList=( "HMM_UW_BIAS_D13_Model"$cell "HMM_UW_D13_Model"$cell )
      fi

      # Predictions Loop
      for pred in $predList
      do

        # Extension Paramers
        extList=( "5" )

        # Extension Loop
        for ext in $extList
        do
            
          # Parameters
          mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev".bed"
          predFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$pred".bed"
          outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/"$pred"_"$ext".bed"

          # Execution
          #extendAndIntersect $ext $ext $mpbsFileName $predFileName $outputFileName
          bsub -J $lab"_"$cell" "$pred"_EIDH" -o $lab"_"$cell" "$pred"_EIDH_out.txt" -e $lab"_"$cell" "$pred"_EIDH_err.txt" -W 1:00 -M 12000 -S 100 -R "select[hpcwork]" ./extendAndIntersect_pipeline.zsh $ext $ext $mpbsFileName $predFileName $outputFileName
          # -P izkf

        done
      done
    done
  done
done


