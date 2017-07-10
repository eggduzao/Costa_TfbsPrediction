#!/bin/zsh

# Lab Parameters
labList=( "DU" "UW" )

# Lab Loop
for lab in $labList
do

  # Cell Parameters
  if [[ $lab == "DU" ]]; then
    cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "Mcf7" "Saos2" )
  elif [[ $lab == "UW" ]]; then
    #cellList=( "HepG2" "Huvec" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" )
    cellList=( "LnCaP" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    # Factor Parameters
    if [[ $cell == "H1hesc" ]]; then
      tfbsList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF" "EGR1" "FOSL1" "GABP" "JUN" "JUND"
                 "MAFK" "MAX" "MYC" "NRF1" "P300" "POU5F1" "RAD21" "REST" "RFX5" "RXRA"
                 "SIX5" "SP1" "SP2" "SP4" "SRF" "TBP" "TCF12" "USF1" "USF2" "YY1" "ZNF143" )
    elif [[ $cell == "HeLaS3" ]]; then
      tfbsList=( "BRCA1" "CEBPB" "CTCF" "E2F4" "E2F6" "ELK1" "FOS" "GABP" "JUN" "JUND"
                 "MAFK" "MAX" "MYC" "NFYA" "NFYB" "NRF1" "P300" "RAD21" "REST" "STAT1" "TBP" "USF2" "ZNF143" )
    elif [[ $cell == "HepG2" ]]; then
      tfbsList=( "ARID3A" "ATF3" "BHLHE40" "BRCA1" "CEBPB" "CTCF" "ELF1" "GABP" "JUN" "JUND" "MAFF" "MAFK" 
                 "MAX" "MYC" "NRF1" "P300" "RAD21" "REST" "RXRA" "SP1" "SP2" "SRF" "TBP" "USF1" "USF2" "YY1" )
    elif [[ $cell == "Huvec" ]]; then
      tfbsList=( "CTCF" "FOS" "GATA2" "JUN" "MAX" "MYC" )
    elif [[ $cell == "K562" ]]; then
      tfbsList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" 
                 "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" "EJUND" "ELF1" "ELK1" "ETS1" "FOS" 
                 "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" "MAFF" "MAFK" "MAX" 
                 "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" 
                 "RFX5" "SIX5" "SMC3" "SP1" "SP2" "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" 
                 "TBP" "THAP1" "TR4" "USF1" "USF2" "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
    elif [[ $cell == "LnCaP" ]]; then
      tfbsList=( "AR_1nmR" "AR_10nmR" "AR_ethl" "AR_R1881" )
    elif [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
      tfbsList=( "GR_withDEX" "GR_woDEX" )
    elif [[ $cell == "Mcf7" ]]; then
      tfbsList=( "ER_0min" "ER_2min" "ER_5min" "ER_10min" "ER_40min" "ER_160min" )
    elif [[ $cell == "Saos2" ]]; then
      tfbsList=( "P53" )
    elif [[ $cell == "H7hesc" ]]; then
      tfbsList=( "UW_Motif_0458" "UW_Motif_0500" )
    fi

    # Factor Loop
    for tfbs in $tfbsList
    do

      # Type Parameters
      #typeList=( "regular" "bias" "naked" )
      typeList=( "naked" )

      # Type Loop
      for type in $typeList
      do

        # Parameters
        if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
          mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/m3134_Evidence/fdr_4/"$tfbs".bed"
        else
          mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/fdr_4/"$tfbs".bed"
        fi

        if [[ $type == "regular" ]]; then
          signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"$lab"_"$cell"/DNase.bw"
          outLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/fdr_4/FOS_"$lab"/"
          outputFileName=$outLoc$tfbs".bed"
        elif [[ $type == "bias" ]]; then
          signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/"$lab"_"$cell"/DNaseBiasCorrected.bw"
          outLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/fdr_4/FOS_"$lab"_BIAS/"
          outputFileName=$outLoc$tfbs".bed"
        elif [[ $type == "naked" ]]; then
          signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/BiasNaked/"$lab"_"$cell"/DNaseBiasCorrected.bw"
          outLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/fdr_4/FOS_"$lab"_NAKED2/"
          outputFileName=$outLoc$tfbs".bed"
        fi
        mkdir -p $outLoc

        # Execution
        bsub -J "FOS2" -o "FOS2_out.txt" -e "FOS2_err.txt" -W 48:00 -M 12000 -S 100 -R "select[hpcwork]" ./applyFOS_pipeline.zsh $mpbsFileName $signalFileName $outputFileName
# -P izkf
      done
    done
  done
done


