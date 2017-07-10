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
    cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "Mcf7" "Saos2" )
  elif [[ $lab == "UW" ]]; then
    cellList=( "HepG2" "Huvec" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    # Predictions Parameters
    if [[ $lab == "DU" ]]; then
      predList=( "HMM_DNase_DU_Model"$cell "HMM_DNase_DU_BIAS_Model"$cell )
    elif [[ $lab == "UW" ]]; then
      predList=( "HMM_DNase_UW_Model"$cell "HMM_DNase_UW_BIAS_Model"$cell  )
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
        if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
          mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"$cell"/TagCount_"$lab".bed"
        elif [[ $cell == "UW" ]]; then
          mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"$cell"/TagCount_"$lab".bed"
        fi
        predFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/"$cell"/"$pred".bed"
        ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/restemp/"
        outputFileName=$ol$cell"/"$pred"_"$ext".bed"
        mkdir -p $ol

        # Execution
        #extendAndIntersect $ext $ext $mpbsFileName $predFileName $outputFileName
        bsub -J $lab"_"$cell"_"$pred"_EIMD" -o $lab"_"$cell"_"$pred"_EIMD_out.txt" -e $lab"_"$cell"_"$pred"_EIMD_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./extendAndIntersectMaxScore_pipeline.zsh $ext $ext $mpbsFileName $predFileName $outputFileName

      done
    done
  done
done


