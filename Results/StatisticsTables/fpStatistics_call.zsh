#!/bin/zsh

# Global Parameters
footLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"
outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TableStatistics/Footprints/"

# Cell Line Parameters
#cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )
cellList=( "H1hesc" )

# Cell Line Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do
        outFile=$outputLocation$cell"_"$ev".tex"
        bsub -J $cell"_"$ev"_TAB" -o $cell"_"$ev"_TAB_out.txt" -e $cell"_"$ev"_TAB_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./fpStatistics_pipeline.zsh $cell $ev $footLocation $outFile
    done
done


