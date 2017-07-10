#!/bin/zsh

# Global Parameters
outExt="png"
allNeg="y"

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )

# Cell Loop
for cell in $cellList
do

    # Parameters
    il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/"
    labelList="global_96_H1hesc,global_96_K562,global_98_H1hesc,global_98_K562,global_99_H1hesc,global_99_K562,local_96_H1hesc,local_96_K562,local_98_H1hesc,local_98_K562,local_99_H1hesc,local_99_K562"
    bedList=$il"global_96/res_merge_ModelH1hesc.bed",$il"global_96/res_merge_ModelK562.bed",$il"global_98/res_merge_ModelH1hesc.bed",$il"global_98/res_merge_ModelK562.bed",$il"global_99/res_merge_ModelH1hesc.bed",$il"global_99/res_merge_ModelK562.bed",$il"local_96/res_merge_ModelH1hesc.bed",$il"local_96/res_merge_ModelK562.bed",$il"local_98/res_merge_ModelH1hesc.bed",$il"local_98/res_merge_ModelK562.bed",$il"local_99/res_merge_ModelH1hesc.bed",$il"local_99/res_merge_ModelK562.bed"
    outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/ROC/"
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
        bsub -J $cell"_ROCG" -o $cell"_ROCG_out.txt" -e $cell"_ROCG_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./rocFromBed_pipeline.zsh $allNeg $outExt $factor $labelList $bedList $outputLocation

    done
done


