#!/bin/zsh

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )

# Cell Loop
for cell in $cellList
do

  # Parameters
  pl="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"

  # Pred list Parameters
  if [[ $cell == "H1hesc" ]]; then

    predList=( "HD_D13_ModelH1hesc" "HD_D139_ModelH1hesc" "HMM_DU_BIAS_D13_ModelH1hesc" )
    labList=( "DU" "DU" "DU" )

  elif [[ $cell == "HeLaS3" ]]; then 

    predList=( "HD_D13_ModelHeLaS3" "HMM_DU_BIAS_D13_ModelHeLaS3" )
    labList=( "DU" "DU" )

  elif [[ $cell == "HepG2" ]]; then 

    predList=( "HD_D13_ModelHepG2" "HMM_DU_BIAS_D13_ModelHepG2" "HMM_UW_D13_ModelHepG2" "HMM_UW_BIAS_D13_ModelHepG2" )
    labList=( "DU" "DU" "UW" "UW" )

  elif [[ $cell == "Huvec" ]]; then 

    predList=( "HD_D13_ModelHuvec" "HMM_DU_BIAS_D13_ModelHuvec" "HMM_UW_D13_ModelHuvec" "HMM_UW_BIAS_D13_ModelHuvec" )
    labList=( "DU" "DU" "UW" "UW" )

  else

    predList=( "HD_D13_ModelK562" "HD_D139_ModelK562" "HMM_DU_BIAS_D13_ModelK562" "HMM_UW_D13_ModelK562" "HMM_UW_BIAS_D13_ModelK562" )
    labList=( "DU" "DU" "DU" "UW" "UW" )

  fi

  # Pred List Loop
  for i in {1..$#predList}
  do

    # Aux Parameters
    pred=$predList[$i]
    lab=$labList[$i]

    # Parameters
    windowLen="200"
    fpFileName=$pl$pred".bed"
    signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"$lab"_"$cell"/DNase.bw"
    outputFileName=$pl$pred"_TC.bed"

    # Execution
    bsub -J $cell"_"$pred"_FTC" -o $cell"_"$pred"_FTC_out.txt" -e $cell"_"$pred"_FTC_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./footprintTagCount_pipeline.zsh  $windowLen $fpFileName $signalFileName $outputFileName

  done
done


