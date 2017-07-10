#!/bin/zsh

inLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_GM12878/Cuellar/DNase/"
csFile="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered.increased.10e6_ALL.ForBw"

bsub -J "JCP" -o "JCP_out.txt" -e "JCP_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./joinCuellarPriorIntoSingleBigwigFile_pipeline.zsh $inLoc $csFile

# Global parameters
#cf="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"
#chromSizesFile="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/hg19.chrom.sizes"
# Cell parameters
#cellList=( "H1hesc" "K562" )
# Cell loop
#for cell in $cellList
#do
    # Signal parameters
    #signalList=( "DNase" "H3K4me1" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
#    signalList=( "H2AZ" "H3K9ac" "H3K27ac" )
    # Signal loop
#    for signal in $signalList
#    do
        # Execution
#        bsub -J $cell"_"$signal"_JCP" -o $cell"_"$signal"_JCP_out.txt" -e $cell"_"$signal"_JCP_err.txt" -W 2:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./joinCuellarPriorIntoSingleBigwigFile_pipeline.zsh $cf$cell"/Cuellar/"$signal"/" $chromSizesFile
#    done
#done


