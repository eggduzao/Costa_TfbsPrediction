#!/bin/zsh

################################################
### NEPH PREDICTIONS
################################################

# Cell Line Parameters
#cellList=( "H1hesc" "K562" )
cellList=( "H1hesc" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Predictions Parameters
        predList=( "Neph" )

        # Predictions Loop
        for pred in $predList
        do

            # Extension Paramers
            extList=( "-5" "0" )

            # Extension Loop
            for ext in $extList
            do
            
                # Parameters
                mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev".bed"
                predFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$pred".bed"
                outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/"$pred"_"$ext".bed"

                # Execution
                bsub -J $cell"_"$ev"_"$pred"_"$ext"_EIN" -o $cell"_"$ev"_"$pred"_"$ext"_EIN_out.txt" -e $cell"_"$ev"_"$pred"_"$ext"_EIN_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./extendAndIntersect_pipeline.zsh $ext $ext $mpbsFileName $predFileName $outputFileName

            done
        done
    done
done


