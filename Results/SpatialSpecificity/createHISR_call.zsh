#!/bin/zsh

# Global Parameters
threshold="100"
hisFolder="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/"
outHisLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/"

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

        #bsub -J $cell"_"$ev"_HISR" -o $cell"_"$ev"_HISR_out.txt" -e $cell"_"$ev"_HISR_err.txt" -W 0:10 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createHISR_pipeline.zsh $threshold $cell $ev $hisFolder $outHisLoc
        ./createHISR_pipeline.zsh $threshold $cell $ev $hisFolder $outHisLoc
    done
done


