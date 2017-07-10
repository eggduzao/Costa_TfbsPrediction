#!/bin/zsh

# Global Parameters
signalLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/"
coordLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"
outLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlopeBias/"

# Cell Line Parameters
#cellList=( "DU_H1hesc" "DU_H7hesc" "DU_HeLaS3" "DU_HepG2" "DU_Huvec" "DU_K562" "UW_HepG2" "UW_Huvec" "UW_K562" "DU_Mcf7" "UW_LnCaP" "UW_m3134_with_DEX" "UW_m3134_wo_DEX" "UW_H7hesc" )
cellList=( "DU_GM12878" )

# Cell Line Loop
for cell in $cellList
do

  # Coordinate Parameters
  coordList=( "hd" )

  # Coordinate Loop
  for coord in $coordList
  do
  
    # Parameters
    slopeWindowSize="9"
    perNorm="98"
    perSlope="98"
    coordFileName=$coordLocation$cell"/"$coord".bed"
    signalFileName=$signalLocation$cell"/DNaseBiasCorrected.bw"
    outputLocation=$outLocation$cell"/"
    mkdir -p $outputLocation

    # Execution
    bsub -J "NSB" -o "NSB_out.txt" -e "NSB_err.txt" -W 24:00 -M 30000 -S 100 -P izkf -R "select[hpcwork]" ./createSignal_pipeline.zsh $slopeWindowSize $perNorm $perSlope $coordFileName $signalFileName $outputLocation

  done
done


