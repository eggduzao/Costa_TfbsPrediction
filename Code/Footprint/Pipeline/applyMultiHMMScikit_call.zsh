#!/bin/zsh

# Lab Parameters
#labList=( "DU" "UW" )
labList=( "UW" )

# Lab Loop
for lab in $labList
do

  # Signal Parameters
  if [[ $lab == "DU" ]]; then
    signalCellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
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
      #hList=( "H2AZ" "H2AZ" "H3K4me1" "H3K4me1" 
      #        "H3K4me3" "H3K4me3" "H3K9ac" "H3K9ac"
      #        "H3K27ac" "H3K27ac" )
      #fList=( "H2AZ_proximal" "H2AZ_proximal_ext" "H3K4me1_distal" "H3K4me1_distal_ext" 
      #        "H3K4me3_proximal" "H3K4me3_proximal_ext" "H3K9ac_proximal" "H3K9ac_proximal_ext"
      #        "H3K27ac_proximal" "H3K27ac_proximal_ext" )
      #oList=( "H2AZ" "H2AZ_ext" "H3K4me1" "H3K4me1_ext" 
      #        "H3K4me3" "H3K4me3_ext" "H3K9ac" "H3K9ac_ext"
      #        "H3K27ac" "H3K27ac_ext" )
      hList=( "H3K4me1" "H3K4me3" )
      fList=( "H3K4me1_distal" "H3K4me3_proximal" )
      oList=( "H3K4me1" "H3K4me3" )

      # Histone Loop
      for i in {1..$#hList}
      do

        # Parameters
        hmmType="8"
        coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"$lab"_"$sigCell"/hd.bed"
        hmmFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/Models_DNaseHistone/"$modCell"/Model/hd/"$fList[$i]".hmm"
        dl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/"$lab"_"$sigCell"/"
        hl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/DU_"$sigCell"/"
        hn=$hList[$i]
        signalList=$dl"DNase_norm.bw,"$dl"DNase_slope.bw,"$hl$hn"_norm.bw,"$hl$hn"_slope.bw"
        ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/"$lab"_HMM/"
        outputFileName=$ol$oList[$i]"_Model"$modCell".bed"
        mkdir -p $ol

        # Execution
        bsub -J $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHS" -o $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHS_out.txt" -e $lab"_"$sigCell"_"$modCell"_"$oList[$i]"_AMHS_err.txt" -W 50:00 -M 12000 -S 100 -R "select[hpcwork]" ./applyMultiHMMScikit_pipeline.zsh $hmmType $coordFileName $hmmFileName $signalList $outputFileName
        # -P izkf

      done
    done
  done
done


