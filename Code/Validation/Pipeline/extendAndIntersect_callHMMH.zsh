#!/bin/zsh

################################################
### HISTONES HMM PREDICTIONS
################################################

# Cell Line Parameters
cellList=( "H1hesc" "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Predictions Parameters
        predList=( "HMMH_D13_ModelH1hesc" "HMMH_D13_ModelK562" "HMMH_D39_ModelH1hesc" "HMMH_D39_ModelK562"
                   "HMMH_D139_ModelH1hesc" "HMMH_D139_ModelK562" )

        # Predictions Loop
        for pred in $predList
        do

            # Extension Paramers
            extList=( "0" )

            # Extension Loop
            for ext in $extList
            do
            
                # Parameters
                mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev".bed"
                predFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$pred".bed"
                outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/"$pred"_"$ext".bed"

                # Execution
                bsub -J $cell"_"$ev"_"$pred"_"$ext"_EIGH" -o $cell"_"$ev"_"$pred"_"$ext"_EIGH_out.txt" -e $cell"_"$ev"_"$pred"_"$ext"_EIGH_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./extendAndIntersect_pipeline.zsh $ext $ext $mpbsFileName $predFileName $outputFileName

            done
        done
    done
done


