#!/bin/zsh

# Cell Line Parameters
signalLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/Counts/"
coordLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/HMM/InputRegions/"
outLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/ATAC/NormSlope/"
cellList=( "K562" )

# Cell Line Loop
for cell in $cellList
do

  # Parameters
  slopeWindowSize="9"
  perNorm="98"
  perSlope="98"
  coordFileName=$coordLocation$cell"/ATAC_HS_Extended.bed"
  signalFileName=$signalLocation$cell"/ATACBO.bw"
  outputLocation=$outLocation$cell"/"
  mkdir -p $outputLocation

  # Execution
  bsub -J "NSA" -o "NSA_out.txt" -e "NSA_err.txt" -W 24:00 -M 30000 -S 100 -P izkf -R "select[hpcwork]" ./createSignal_pipeline.zsh $slopeWindowSize $perNorm $perSlope $coordFileName $signalFileName $outputLocation

done


