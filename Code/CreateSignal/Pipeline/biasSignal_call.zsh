#!/bin/zsh

p="/work/eg474423/eg474423_Projects/trunk"

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
    #fBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/"$lab"_"$cell"_6_F.txt"
    #rBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/"$lab"_"$cell"_6_R.txt"
    fBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/table/hs_DUHS_Gm12878_6_F.txt"
    rBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/table/hs_DUHS_Gm12878_6_R.txt"
    coordFileName=$p"/TfbsPrediction/Results/HMM/InputRegions/"$labLabel"_"$cell"/dnase_HS_regions.bed"
    bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/"$lab"/"$cell"/DNase.bam"
    if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes"
    else
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
    fi
    outLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/"$labLabel"_"$cell"/"
    outputFileName=$outLoc"DNaseBiasCorrected.wig"
    mkdir -p $outLoc

    # Execution
    bsub -J "BS" -o "BS_out.txt" -e "BS_err.txt" -W 48:00 -M 40000 -S 100 -P izkf -R "select[hpcwork]" ./biasSignal_pipeline.zsh $fBiasFileName $rBiasFileName $coordFileName $bamFileName $fastaFileName $csFileName $outputFileName
  done
done


