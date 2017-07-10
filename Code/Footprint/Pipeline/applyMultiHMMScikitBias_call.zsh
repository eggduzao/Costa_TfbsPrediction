#!/bin/zsh

# Lab Parameters
labList=( "DU" "UW" )
labList=( "DU" )

# Lab Loop
for lab in $labList
do

  # Signal Parameters
  if [[ $lab == "DU" ]]; then
    signalCellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
    signalCellList=( "HeLaS3" )
  elif [[ $lab == "UW" ]]; then
    signalCellList=( "HepG2" "Huvec" "K562" )
  fi

  # Signal Loop
  for sigCell in $signalCellList
  do

    # Model Parameters
    #if [[ $lab == "DU" ]]; then
      #modelCellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
    #elif [[ $lab == "UW" ]]; then
      #modelCellList=( "HepG2" "Huvec" "K562" )
    #fi
    modelCellList=( $sigCell )

    # Model Loop
    for modCell in $modelCellList
    do

      # Histone Parameters
      hList=( "H3K4me1" "H3K4me3" )
      fList=( "H3K4me1_distal" "H3K4me3_proximal" )
      oList=( "H3K4me1" "H3K4me3" )

      hList=( "H3K4me3" )
      fList=( "H3K4me3_proximal" )
      oList=( "H3K4me3" )

      # Histone Loop
      for i in {1..$#hList}
      do

        # Parameters
        hmmType="8"
        coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"$lab"_"$sigCell"/hd.bed"
        hmmFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/Models_DNaseHistone/"$modCell"/Model/bias/"$fList[$i]".hmm"
        dl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlopeBias/"$lab"_"$sigCell"/"
        hl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/DU_"$sigCell"/"
        hn=$hList[$i]
        signalList=$dl"DNaseBiasCorrected_norm.bw,"$dl"DNaseBiasCorrected_slope.bw,"$hl$hn"_norm.bw,"$hl$hn"_slope.bw"
        ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/HMM_"$lab"_BIAS/"
        outputFileName=$ol$oList[$i]"_Model"$modCell".bed"
        mkdir -p $ol

        # Execution
        bsub -J $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHSB" -o $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHSB_out.txt" -e $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHSB_err.txt" -W 80:00 -M 24000 -S 100 -R "select[hpcwork]" ./applyMultiHMMScikit_pipeline.zsh $hmmType $coordFileName $hmmFileName $signalList $outputFileName
        # -P izkf

      done
    done
  done
done


