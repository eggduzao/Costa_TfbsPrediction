#!/bin/zsh

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )

# Cell Loop
for cell in $cellList
do

    # Grid Parameters
    normTypeList=( "global" "local" )
    perList=( "96" "98" "99" )

    # Grid Loop
    for normType in $normTypeList
    do
        for per in $perList
        do

            # Predictions Parameters
            predList=( "fp_H3K4me1_ModelH1hesc" "fp_H3K4me1_ModelK562" "fp_H3K4me3_ModelH1hesc" 
                       "fp_H3K4me3_ModelK562" "fp_merge_ModelH1hesc" "fp_merge_ModelK562" )

            # Predictions Loop
            for pred in $predList
            do
            
                # Parameters
                leftExt="20"
                rightExt="20"
                mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBS/"$cell"_Evidence/evidence_fdr_4_sort_chr1.bed"
                predFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/"$normType"_"$per"/"$pred".bed"
                outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/GridTest/"$cell"/"$normType"_"$per"/RES/"
                mkdir -p $outputLocation

                # Execution
                bsub -J $cell"_EIGR" -o $cell"_EIGR_out.txt" -e $cell"_EIGR_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./extendAndIntersect_pipeline.zsh $leftExt $rightExt $mpbsFileName $predFileName $outputLocation

            done
        done
    done
done


