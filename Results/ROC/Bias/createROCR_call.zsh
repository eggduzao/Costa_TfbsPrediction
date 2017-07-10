#!/bin/zsh

# Global Parameters
rocFolder="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/"
outRocLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/Bias/"

# Cell Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )

# Cell Loop
for cell in $cellList
do
  #bsub -J $cell"_"$ev"_ROCR" -o $cell"_"$ev"_ROCR_out.txt" -e $cell"_"$ev"_ROCR_err.txt" -W 0:20 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createROCR_pipeline.zsh $cell $rocFolder $outRocLoc
  mkdir -p $outRocLoc$cell
  python ./createROCR.py $cell $rocFolder $outRocLoc
done


