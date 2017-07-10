#!/bin/zsh

# Cell Line Parameters
#cellList=( "H1hesc" "K562" )
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # TF parameters
        if [[ $cell == "H1hesc" ]]; then
            tfbsList=(  )
        else
            tfbsList=( "IRF1" "ZNF263" )
        fi

        # TF Loop
        for i in {1..$#tfbsList}
        do

            # Parameters
            mpbsName=$tfbsList[$i]
            mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev"/"$mpbsName".bed"
            dampFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/CentipedeEstimation/defaultAUC.txt"
            priorFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/Centipede/"$mpbsName"_PRIOR.txt"
            dnaseFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/Centipede/"$mpbsName"_DNASE.txt"
            outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/Pique/"$mpbsName"_def.bed"

            # Execution
            bsub -J $cell"_"$ev"_"$mpbsName"_ACD" -o $cell"_"$ev"_"$mpbsName"_ACD_out.txt" -e $cell"_"$ev"_"$mpbsName"_ACD_err.txt" -W 100:00 -M 100000 -S 100 -P izkf -R "select[hpcwork]" ./applyCentipede_pipeline.zsh $mpbsName $mpbsFileName $dampFileName $priorFileName $dnaseFileName $outputFileName

        done
    done
done


