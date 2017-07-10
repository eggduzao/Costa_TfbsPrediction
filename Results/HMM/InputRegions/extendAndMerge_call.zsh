#!/bin/zsh

# Lab Parameters
labList=( "DNase" "DNaseUW" )
labLabels=( "DU" "UW" )
labList=( "DNase_DU" )
labLabels=( "DU" )

# Lab Loop
for l in {1..$#labList}
do

  # Cell Parameters
  lab=$labList[$l]
  labLabel=$labLabels[$l]
  if [[ $labLabel == "DU" ]]; then
    #cellList=( "H1hesc" "H7hesc" "HeLaS3" "HepG2" "Huvec" "K562" "Mcf7" )
    cellList=( "GM12878" )
  elif [[ $labLabel == "UW" ]]; then
    #cellList=( "HepG2" "Huvec" "K562" "IMR90" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" "m3134_with_DEX" )
    cellList=( "H7hesc" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    leftExt="5000"
    rightExt="5000"
    #coordList="/hpcwork/izkf/projects/TfbsPrediction/Data/"$lab"/"$cell"/DNasePeaks.bed,/hpcwork/izkf/projects/TfbsPrediction/Results/HistonePeaks/"$cell"/H3K4me1/H3K4me1_peaks.bed,/hpcwork/izkf/projects/TfbsPrediction/Results/HistonePeaks/"$cell"/H3K4me3/H3K4me3_peaks.bed"
    coordList="/hpcwork/izkf/projects/TfbsPrediction/Data/"$lab"/"$cell"/DNasePeaks.bed"
    outputFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"$labLabel"_"$cell"/dnase_HS_regions.bed"

    extendAndMerge $leftExt $rightExt $coordList $outputFileName
    #simpleBedCoverage $outputFileName

  done
done


