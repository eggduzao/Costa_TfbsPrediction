#!/bin/zsh

# Global Parameters
normVec="y"
windowLen="10000"

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )

# Cell Loop
for cell in $cellList
do

    # Parameters
    mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBS/"$cell"_Evidence/evidence_fdr_4_sort_chr1.bed"
    il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/"
    predLabelList="global_96_H1hesc,global_96_K562,global_98_H1hesc,global_98_K562,global_99_H1hesc,global_99_K562,local_96_H1hesc,local_96_K562,local_98_H1hesc,local_98_K562,local_99_H1hesc,local_99_K562"
    predFileNameList=$il"global_96/ext_merge_ModelH1hesc.bed",$il"global_96/ext_merge_ModelK562.bed",$il"global_98/ext_merge_ModelH1hesc.bed",$il"global_98/ext_merge_ModelK562.bed",$il"global_99/ext_merge_ModelH1hesc.bed",$il"global_99/ext_merge_ModelK562.bed",$il"local_96/ext_merge_ModelH1hesc.bed",$il"local_96/ext_merge_ModelK562.bed",$il"local_98/ext_merge_ModelH1hesc.bed",$il"local_98/ext_merge_ModelK562.bed",$il"local_99/ext_merge_ModelH1hesc.bed",$il"local_99/ext_merge_ModelK562.bed"
    outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/BOX/"
    mkdir -p $outputLocation

    # Cell specific parameters
    if [[ $cell == "H1hesc" ]]; then
        factorList=( "CTCF_MA0139" "GABP_M00108" "JUN_M00036" "MAX_M00118" "REST_M00256" "SRF_MA0083" "YY1_M00069" )
    elif [[ $cell == "HeLaS3" ]]; then
        factorList=( "GABP_M00108" "JUN_M00036" )
    elif [[ $cell == "HepG2" ]]; then
        factorList=( "GABP_M00108" "JUN_M00036" )
    else
        factorList=( "CTCF_MA0139" "GABP_M00108" "GATA1_M00128" "GATA2_M00348" "GCN4_M00038" "JUN_M00036" "MAX_M00118" "NFE2_M00037" "REST_M00256" "SRF_MA0083" "YY1_M00069" )
    fi

    # Factor Loop
    for factor in $factorList
    do
        bsub -J $cell"_HISG" -o $cell"_HISG_out.txt" -e $cell"_HISG_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./predictionHistogram_pipeline.zsh $normVec $windowLen $factor $mpbsFileName $predLabelList $predFileNameList $outputLocation
    done

    # All factors execution
    bsub -J $cell"_HISG" -o $cell"_HISG_out.txt" -e $cell"_HISG_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./predictionHistogram_pipeline.zsh $normVec $windowLen "all" $mpbsFileName $predLabelList $predFileNameList $outputLocation

done


