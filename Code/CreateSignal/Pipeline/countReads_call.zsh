#!/bin/zsh

# Lab Parameters
#labList=( "DNase" "DNaseUW" )
#labLabels=( "DU" "UW" )
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
    cellList=( "H7hesc" "HepG2" "Huvec" "IMR90" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    # Parameters
    leftInc="0"
    rightInc="1"
    bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/"$lab"/"$cell"/DNase.bam"
    if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes"
    else
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
    fi
    ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"$labLabel"_"$cell"/"
    outputFileName=$ol"DNaseNeg.wig"
    mkdir -p $ol
    
    # Execution
    bsub -J "CR" -o "CR_out.txt" -e "CR_err.txt" -W 48:00 -M 60000 -S 100 -P izkf -R "select[hpcwork]" ./countReads_pipeline.zsh $leftInc $rightInc $bamFileName $csFileName $outputFileName "N"

  done
done


