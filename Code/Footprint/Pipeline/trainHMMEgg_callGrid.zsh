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
            sList=( "H3K4me1" "H3K4me1" "H3K4me3" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
            fList=( "H3K4me1_distal" "H3K4me1_proximal" "H3K4me3_distal" "H3K4me3_proximal" "H2AZ_proximal" "H3K9ac_proximal" "H3K27ac_proximal" )

            # Signal Loop
            for i in {1..$#fList}
            do
            
                # Parameters
                clearNonExistingTrans="y"
                dataModeList="n,n,n,n"
                starterModel="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/blank.hmm"
                fl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Annotation/stt/"
                trainSet=$fl$fList[$i]".stt"
                sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/InputSignal/"$normType"_"$per"/"
                hn=$sList[$i]
                signalList=$sl"DNase_norm.bw,"$sl"DNase_slope.bw,"$sl$hn"_norm.bw,"$sl$hn"_slope.bw"
                outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/"$normType"_"$per"/"
                mkdir -p $outputLocation

                # Execution
                trainHMMEgg $clearNonExistingTrans $dataModeList $starterModel $trainSet $signalList $outputLocation
                mv $outputLocation"model.hmm" $outputLocation$fList[$i]".hmm"

            done
        done
    done
done


