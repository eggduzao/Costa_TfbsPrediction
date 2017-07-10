#!/bin/zsh

# Global Parameters
rocFolder="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/"
outRocLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/"

# Cell Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "LnCaP" "m3134_with_DEX" "m3134_wo_DEX" "Mcf7" "Saos2" )
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do
  #bsub -J $cell"_"$ev"_ROCR" -o $cell"_"$ev"_ROCR_out.txt" -e $cell"_"$ev"_ROCR_err.txt" -W 0:20 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createROCR_pipeline.zsh $cell $rocFolder $outRocLoc
  mkdir -p $outRocLoc$cell
  python ./createROCR.py $cell $rocFolder $outRocLoc
done


