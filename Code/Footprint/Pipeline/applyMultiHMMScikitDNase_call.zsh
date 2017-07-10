#!/bin/zsh

# Lab Parameters
# labList=( "DU" "UW" )
labList=( "DU" )

# Lab Loop
for lab in $labList
do

  # Signal Parameters
  if [[ $lab == "DU" ]]; then
    #signalCellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "Mcf7" )
    signalCellList=( "GM12878" )
  elif [[ $lab == "UW" ]]; then
    signalCellList=( "H7hesc" "HepG2" "Huvec" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" )
  fi

  # Signal Loop
  for sigCell in $signalCellList
  do

    modelCellList=( $sigCell )

    # Model Loop
    for modCell in $modelCellList
    do

      # Type Parameters
      #typeList=( "regular" "bias" "naked" )
      typeList=( "naked" )

      # Type Loop
      for type in $typeList
      do

        # Parameters
        coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"$lab"_"$sigCell"/hd.bed"
        hmmFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/Models_DNase/"$modCell"/Model/"$type".hmm"
        if [[ $type == "regular" ]]; then
          sl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/"$lab"_"$sigCell"/"
          signalList=$sl"DNase_norm.bw,"$sl"DNase_slope.bw"
          ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/HMM_DNase_"$lab"/"
          outputFileName=$ol"DNase_Model"$modCell".bed"
          mkdir -p $ol
        elif [[ $type == "bias" ]]; then
          sl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlopeBias/"$lab"_"$sigCell"/"
          signalList=$sl"DNaseBiasCorrected_norm.bw,"$sl"DNaseBiasCorrected_slope.bw"
          ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/HMM_DNase_"$lab"_BIAS/"
          outputFileName=$ol"DNase_Model"$modCell".bed"
          mkdir -p $ol
        elif [[ $type == "naked" ]]; then
          sl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlopeBiasNaked/"$lab"_"$sigCell"/"
          signalList=$sl"DNaseBiasCorrected_norm.bw,"$sl"DNaseBiasCorrected_slope.bw"
          ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/HMM_DNase_"$lab"_NAKED/"
          outputFileName=$ol"DNase_Model"$modCell".bed"
          mkdir -p $ol
        fi

        # Execution
        bsub -J $lab"_"$sigCell"_"$type"_AMSD" -o $lab"_"$sigCell"_"$type"_AMSD_out.txt" -e $lab"_"$sigCell"_"$type"_AMSD_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyMultiHMMScikitDNase_pipeline.zsh $coordFileName $hmmFileName $signalList $outputFileName

      done
    done
  done
done


