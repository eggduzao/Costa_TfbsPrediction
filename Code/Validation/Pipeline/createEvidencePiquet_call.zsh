#!/bin/zsh

# Parameters
peakExt=100
negExt=400
tfbsLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/"
mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBS/All/bitscore_log.bed"
outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBS/"

# Variations
cellList=( "HeLaS3" "HepG2" )
#tfList=( "CTCF_MA0139" "GABP_M00108" "GATA1_M00128" "GATA2_M00348" "GCN4_M00038" "JUN_M00036" 
#         "MAX_M00118" "NFE2_M00037" "REST_M00256" "SRF_MA0083" "YY1_M00069" )
tfList=( "GABP_M00108" "JUN_M00036" )

# Execution
for cell in $cellList
do
    outLoc=$outputLocation$cell"_Evidence/bitscore_log/"
    mkdir -p $outLoc
    for tf in $tfList
    do
        tfSmall=$tf[1,-8]
        bsub -J $cell"_"$tfSmall"_CPE" -o $cell"_"$tfSmall"_CPE_out.txt" -e $cell"_"$tfSmall"_CPE_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createEvidencePiquet_pipeline.zsh $peakExt $negExt $tf $tfbsLocation$cell"/"$tfSmall"/"$tfSmall"_summits.bed" $tfbsLocation$cell"/"$tfSmall"/"$tfSmall"_treat_signal.bw" $tfbsLocation$cell"/"$tfSmall"/"$tfSmall"_control_signal.bw" $mpbsFileName $outLoc
    done
done


