#!/bin/zsh

# Cell Parameters
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do

  # Parameters
  fBiasFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/KmerBias/KmerResults/Tables/HS/"$cell"_8_F.txt"
  rBiasFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/KmerBias/KmerResults/Tables/HS/"$cell"_8_R.txt"
  coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/HMM/InputRegions/"$cell"/ATAC_HS_Extended.bed"
  bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/"$cell"/ATAC.bam"
  fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
  csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
  outLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/Bias/"$cell"/"
  mkdir -p $outLoc

  # Execution ATAC
  extParam="4,-5,0,1"
  outputFileName=$outLoc"ATACBiasCorrected.wig"
  bsub -J "BSA" -o "BSA_out.txt" -e "BSA_err.txt" -W 120:00 -M 40000 -S 100 -P izkf -R "select[hpcwork]" ./biasSignalATAC_pipeline.zsh $extParam $fBiasFileName $rBiasFileName $coordFileName $bamFileName $fastaFileName $csFileName $outputFileName

  # Execution ATAC-BO
  extParam="0,0,0,9"
  outputFileName=$outLoc"ATACBOBiasCorrected.wig"
  bsub -J "BSA" -o "BSA_out.txt" -e "BSA_err.txt" -W 120:00 -M 40000 -S 100 -P izkf -R "select[hpcwork]" ./biasSignalATAC_pipeline.zsh $extParam $fBiasFileName $rBiasFileName $coordFileName $bamFileName $fastaFileName $csFileName $outputFileName

done


