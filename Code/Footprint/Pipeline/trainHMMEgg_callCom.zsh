#!/bin/zsh

# Cell Line Parameters
#cellList=( "H1hesc" "K562" )
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do

    # Grid Parameters
    #normTypeList=( "global" "local" )
    #perList=( "96" "98" "99" )
    normTypeList=( "local" )
    perList=( "98" )

    # Grid Loop
    for normType in $normTypeList
    do
        for per in $perList
        do
    
            # Signal Parameters
            sList=( "H3K4me1" "H3K4me1" "H3K4me3" "H3K4me3" 
                    "H2AZ" "H3K9ac" "H3K27ac" )
            fList=( "H3K4me1_distal_com" "H3K4me1_proximal_com" "H3K4me3_distal_com" "H3K4me3_proximal_com" 
                    "H2AZ_proximal_com" "H3K9ac_proximal_com" "H3K27ac_proximal_com" )

            # Signal Loop
            for i in {1..$#fList}
            do
            
                # Parameters
                clearNonExistingTrans="y"
                dataModeList="n,n"
                starterModel="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/blank_com.hmm"
                fl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Annotation/stt/"
                trainSet=$fl$fList[$i]".stt"
                sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/InputSignal/"$normType"_"$per"/"
                hn=$sList[$i]
                signalList=$sl"DNase_norm.bw,"$sl$hn"_norm.bw"
                outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/"$normType"_"$per"/"
                mkdir -p $outputLocation

                # Execution
                trainHMMEgg $clearNonExistingTrans $dataModeList $starterModel $trainSet $signalList $outputLocation
                mv $outputLocation"model.hmm" $outputLocation$fList[$i]".hmm"

            done
        done
    done
done


