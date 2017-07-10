#!/bin/zsh

# Cell Line Parameters
sigCellList=( "H1hesc" "K562" )
estCellList=( "K562" "H1hesc" )

# Cell Loop
for j in {1..$#sigCellList}
do
   
    sigCell=$sigCellList[$j]
    estCell=$estCellList[$j]

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # TF parameters
        if [[ $sigCell == "H1hesc" ]]; then
            tfbsList=( "POU5F1" "SP1" )
        else
            tfbsList=( "ARID3A" "IRF1" "MEF2A" "PU1" "STAT2" "ZNF263" )
        fi

        # TF Loop
        for i in {1..$#tfbsList}
        do

            # Parameters
            mpbsName=$tfbsList[$i]
            mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$sigCell"_Evidence/"$ev"/"$mpbsName".bed"
            dampFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/CentipedeEstimation/"$sigCell"/AUC_CellSpecific_"$estCell".txt"
            priorFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$sigCell"/Centipede/"$mpbsName"_PRIOR.txt"
            dnaseFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$sigCell"/Centipede/"$mpbsName"_DNASE.txt"
            outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"$sigCell"/"$ev"/Pique/"$mpbsName"_ces.bed"

            # Execution
            bsub -J $sigCell"_"$mpbsName"_ACC" -o $sigCell"_"$mpbsName"_ACC_out.txt" -e $sigCell"_"$mpbsName"_ACC_err.txt" -W 100:00 -M 100000 -S 100 -P izkf -R "select[hpcwork]" ./applyCentipede_pipeline.zsh $mpbsName $mpbsFileName $dampFileName $priorFileName $dnaseFileName $outputFileName

        done
    done
done


