#!/bin/zsh

# Input Location Parameters
il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"
inputList=( $il"H1hesc/fdr_4/Pique/" $il"K562/fdr_4/Pique/" )

# Input Location Loop
for inputLocation in $inputList
do

    # Signal Type Parameters
    #sigList=( "def" "est" "ces" )
    sigList=( "ces" )

    # Signal Type Loop
    for sigType in $sigList
    do

        # Execution
        #bsub -J "FPR" -o "FPR_out.txt" -e "FPR_err.txt" -W 1:00 -M 4000 -S 100 -P izkf -R "select[hpcwork]" 
        ./fixPiqueResults_pipeline.zsh $sigType $inputLocation

    done
done


