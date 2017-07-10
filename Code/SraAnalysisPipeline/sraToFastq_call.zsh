#!/bin/zsh

# Cell Loop
cellList=( "NakedK562" "NakedMcf7" )
for cell in $cellList
do
  inLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/"$cell"/"
  inList=( $inLoc*".sra" )
  for fn in $inList
  do
    bsub -J "S2F" -o "S2F_out.txt" -e "S2F_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./sraToFastq_pipeline.zsh $fn $inLoc
  done
done


