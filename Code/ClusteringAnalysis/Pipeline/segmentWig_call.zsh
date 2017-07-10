#!/bin/zsh

# Parameters
anType="all"
windowSize="1000000"
normFactorList="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
helpFileName="/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/RegulatoryAnalysisTools/data/hg19/chrom.sizes"
wl1="/hpcwork/izkf/projects/stemaging/Data/histone-fibroblast/"
wl2="/hpcwork/izkf/projects/stemaging/fibroblast/single_peaks/"
wigList=$wl1"Histone_H3K4me1/H3K4me1_treat_signal.bw,"$wl1"Histone_H3K4me3/H3K4me3_treat_signal.bw,"$wl1"Histone_H3K9me3/H3K9me3_treat_signal.bw,"$wl1"Histone_H3K27me3/H3K27me3_treat_signal.bw,"$wl1"Histone_H3K36me3/H3K36me3_treat_signal.bw,"$wl2"patient5_filtered_srt_no_duplicates/patient5_filtered_srt_no_duplicates_treat_signal.bw,"$wl2"patient6_filtered_srt_no_duplicates/patient6_filtered_srt_no_duplicates_treat_signal.bw,"$wl2"patient7_filtered_srt_no_duplicates/patient7_filtered_srt_no_duplicates_treat_signal.bw,"$wl2"patient8_filtered_srt_no_duplicates/patient8_filtered_srt_no_duplicates_treat_signal.bw"
outputLocation="/hpcwork/izkf/projects/stemaging/edu/wigSummary/"

# Execution
bsub -J "SEG" -o "SEG_out.txt" -e "SEG_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./segmentWig_pipeline.zsh $anType $windowSize $normFactorList $helpFileName $wigList $outputLocation


