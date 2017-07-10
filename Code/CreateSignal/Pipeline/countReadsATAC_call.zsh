#!/bin/zsh

# Cell Parameters
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do

  # Parameters
  bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/"$cell"/ATAC.bam"
  csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
  ol="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/Counts/"$cell"/"
  outputFileName=$ol"ATAC.wig"
  outputFileNameBO=$ol"ATACBO.wig"
  mkdir -p $ol

  # Execution ATAC
  bsub -J "CRA" -o "CRA_out.txt" -e "CRA_err.txt" -W 48:00 -M 60000 -S 100 -P izkf -R "select[hpcwork]" ./countReadsATAC_pipeline.zsh "4,-5,0,1" $bamFileName $csFileName $outputFileName

  # Execution ATAC Base Overlap
  bsub -J "CRA" -o "CRA_out.txt" -e "CRA_err.txt" -W 48:00 -M 60000 -S 100 -P izkf -R "select[hpcwork]" ./countReadsATAC_pipeline.zsh "0,0,0,9" $bamFileName $csFileName $outputFileNameBO

done


