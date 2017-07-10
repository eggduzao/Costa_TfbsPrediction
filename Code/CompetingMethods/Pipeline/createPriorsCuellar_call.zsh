#!/bin/zsh

# Global parameters
chromSizesFile="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
alpha="1.0"
beta="10000"

# Cell Line Parameters
#cellList=( "H1hesc" "K562" "GM12878" )
cellList=( "GM12878" )

# Cell Line Loop
for cell in $cellList
do

    # Signal Parameters
    sigLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"$cell"/Cuellar/"
    #signalList=( "DNase" "H3K4me1" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
    signalList=( "DNase" )

    # Signal Loop
    for signal in $signalList
    do

        # Execution
        bsub -J $cell"_"$signal"_CPC" -o $cell"_"$signal"_CPC_out.txt" -e $cell"_"$signal"_CPC_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createPriorsCuellar_pipeline.zsh $alpha $beta $chromSizesFile $sigLoc$signal".bw" $sigLoc

    done
done


