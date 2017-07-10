#!/bin/zsh

cellList=( "K562" )

# Signal Loop
for cell in $cellList
do

  # Type Parameters
  typeList=( "regular" "bias" )

  # Type Loop
  for type in $typeList
  do

    # Parameters
    coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/HMM/InputRegions/"$cell"/ATAC_HS_Extended.bed"
    hmmFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/HMM/Models_ATAC/"$cell"/Model/"$type".hmm"
    if [[ $type == "regular" ]]; then
      sl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/NormSlope/"$cell"/"
      signalList=$sl"ATACBO_norm.bw,"$sl"ATACBO_slope.bw"
      ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ATAC/Predictions/"$cell"/HINT/"
    elif [[ $type == "bias" ]]; then
      sl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/NormSlopeBias/"$cell"/"
      signalList=$sl"ATACBOBiasCorrected_norm.bw,"$sl"ATACBOBiasCorrected_slope.bw"
      ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ATAC/Predictions/"$cell"/HINT-BC/"
    fi
    outputFileName=$ol"footprints.bed"
    mkdir -p $ol

    # Execution
    bsub -J $cell"_"$type"_AMSA" -o $cell"_"$type"_AMSA_out.txt" -e $cell"_"$type"_AMSA_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyMultiHMMScikitDNase_pipeline.zsh $coordFileName $hmmFileName $signalList $outputFileName

  done
done


