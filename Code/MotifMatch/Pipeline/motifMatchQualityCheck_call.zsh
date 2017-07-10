#!/bin/zsh

# Parameters
specIntList="1,7"
pwmFileName="/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/RegulatoryAnalysisTools/data/PWM/MA0147.1.Myc.pwm"
inLoc="/work/eg474423/ig440396_dendriticcells/local/myc/MotifStatistics/selected_fdr_0.001_diffwindow/250/"
fastaFileName="/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/RegulatoryAnalysisTools/data/hg19/genome.fa"
outputLocation="/work/eg474423/ig440396_dendriticcells/local/myc/MotifQuality/"

# Variation
labelList=( "K562_myc" "Nhek_myc" )
inList=( "K562_myc_summits/K562_myc.bed" "Nhek_cMyc/Nhek_myc.bed" )

# Execution
for i in {1..$#inList}
do
    bsub -J $labelList[$i] -o $labelList[$i]"_MQ_out.txt" -e $labelList[$i]"_MQ_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./motifMatchQualityCheck_pipeline.zsh $specIntList $pwmFileName $inLoc$inList[$i] $fastaFileName $outputLocation
done


