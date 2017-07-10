#!/bin/zsh

################################################
### PEAKS PREDICTIONS
################################################

# Cell Line Parameters
#cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
cellList=( "HepG2" "Huvec" "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Predictions Parameters
        #predList=( "DNase_DU" "D13_DU" "DNase_UW" "D13_UW" )
        predList=( "DNase_UW" "D13_UW" )

        # Predictions Loop
        for pred in $predList
        do
            
            # Parameters
            mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev".bed"
            predFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$pred".bed"
            outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/"$pred".bed"

            # Execution
            #bsub -J $cell"_"$ev"_"$pred"_EIP" -o $cell"_"$ev"_"$pred"_EIP_out.txt" -e $cell"_"$ev"_"$pred"_EIP_err.txt" -W 5:00 -M 12000 -S 100 -sp 10 -P izkf -R "select[hpcwork]"
            #./extendAndIntersect_pipeline.zsh "0" "0" $mpbsFileName $predFileName $outputFileName
            echo $cell" "$ev" "$pred
            extendAndIntersect "0" "0" $mpbsFileName $predFileName $outputFileName

        done
    done
done


