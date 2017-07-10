#!/bin/zsh

# Global Parameters
rocFolder="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/"
texFolder="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/StatisticsTables/Footprints/"
outRocLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/"

# Cell Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        #bsub -J $cell"_"$ev"_ROCR" -o $cell"_"$ev"_ROCR_out.txt" -e $cell"_"$ev"_ROCR_err.txt" -W 0:20 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createROCR_pipeline.zsh $cell $ev $rocFolder $texFolder $outRocLoc
        ./createROCR_pipeline.zsh $cell $ev $rocFolder $texFolder $outRocLoc

    done
done


