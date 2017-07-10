#!/bin/zsh

# Fix
#p="/home/egg/Projects"
p="/work/eg474423/eg474423_Projects/trunk"

# Lab Parameters
#labList=( "DNase" "DNaseUW" )
#labLabels=( "DU" "UW" )
labList=( "DNaseUW" )
labLabels=( "UW" )

# Lab Loop
for l in {1..$#labList}
do

  # Cell Parameters
  lab=$labList[$l]
  labLabel=$labLabels[$l]
  if [[ $labLabel == "DU" ]]; then
    cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "Mcf7" )
  elif [[ $labLabel == "UW" ]]; then
    #cellList=( "HepG2" "Huvec" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" )
    cellList=( "H7hesc" )
  fi

  # Cell Loop
  for cell in $cellList
  do

    # Parameters
    fBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/"$lab"_"$cell"_6_F.txt"
    rBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/"$lab"_"$cell"_6_R.txt"
    bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/"$lab"/"$cell"/DNase.bam"
    if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes"
      tfbsLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/m3134_Evidence/fdr_4/"
    else
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
      tfbsLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/fdr_4/"
    fi
    outputLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"$labLabel"_"$cell"/"
    pwmLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"$labLabel"_"$cell"/PWM/"

    mkdir -p $outputLocation
    mkdir -p $pwmLocation

    # Execution
    bsub -J "EP" -o "EP_out.txt" -e "EP_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./evaluatePearson_pipeline.zsh $fBiasFileName $rBiasFileName $bamFileName $fastaFileName $csFileName $tfbsLocation $outputLocation $pwmLocation "0"

  done
done

#################
# UW Motifs
#################

# Parameters
fBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_H7hesc_6_F.txt"
rBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_H7hesc_6_R.txt"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/H7hesc/DNase.bam"
fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes"
tfbsLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/denovo_Evidence/fdr_4/"
outputLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/UW_denovo/"
pwmLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/UW_denovo/PWM/"
mkdir -p $outputLocation
mkdir -p $pwmLocation

# Execution
#bsub -J "EPU" -o "EPU_out.txt" -e "EPU_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./evaluatePearson_pipeline.zsh $fBiasFileName $rBiasFileName $bamFileName $fastaFileName $csFileName $tfbsLocation $outputLocation $pwmLocation "0"


#################
# P53 Motif in Saos2 intersect K562
#################

# Parameters
fBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_K562_6_F.txt"
rBiasFileName=$p"/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_K562_6_R.txt"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/K562/DNase.bam"
fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes"
tfbsLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/Saos2_Evidence/fdr_4/"
outputLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/DU_Saos2/"
pwmLocation=$p"/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/DU_Saos2/PWM/"
mkdir -p $outputLocation
mkdir -p $pwmLocation

# Execution
#bsub -J "EPS" -o "EPS_out.txt" -e "EPS_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./evaluatePearson_pipeline.zsh $fBiasFileName $rBiasFileName $bamFileName $fastaFileName $csFileName $tfbsLocation $outputLocation $pwmLocation "0"


