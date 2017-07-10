#!/bin/zsh

###################################################################################################
# A. Operations:
#    1. Remove _prior string
#    2. Add Cuellar type
#    3. Copy file to predictions folder
###################################################################################################

# Parameters
#il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"
#inputList=( $il*"/Cuellar5/" )

# Execution
#for inputLocation in $inputList
#do
    #bsub -J "FCR" -o "FCR_out.txt" -e "FCR_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" 
#    ./fixCuellarResults_pipeline.zsh $inputLocation
#done

###################################################################################################
# B. Operations:
#    1. Concatenate all results from Results folder
###################################################################################################

# Input Location Parameters
il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"
inputList=( $il"H1hesc/fdr_4/Cuellar/" $il"K562/fdr_4/Cuellar/" )
outputList=( $il"H1hesc/fdr_4/" $il"K562/fdr_4/" )

# Input Location Loop
for i in {1..$#inputList}
do

    inputLocation=$inputList[$i]
    outputLocation=$outputList[$i]

    # Signal Type Parameters
    sigList=( "Cuellar5_D13" "Cuellar5_D39" "Cuellar5_D139" )

    # Signal Type Loop
    for sigType in $sigList
    do

        # Execution
        #bsub -J "FPR" -o "FPR_out.txt" -e "FPR_err.txt" -W 1:00 -M 4000 -S 100 -P izkf -R "select[hpcwork]"
        cat $inputLocation$sigType"_"*".bed" > $outputLocation$sigType".bed"

    done
done


