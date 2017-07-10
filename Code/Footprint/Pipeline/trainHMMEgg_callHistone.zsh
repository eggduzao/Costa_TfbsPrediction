#!/bin/zsh

# Cell Line Parameters
cellList=( "H1hesc" "K562" )

# Cell Loop
for cell in $cellList
do

    # Model Parameters
    modelList=( "hd" )

    # Grid Loop
    for model in $modelList
    do
    
        # Signal Parameters
        sList=( "H3K9ac" )
        fList=( "H3K9ac_proximal" )

        # Signal Loop
        for i in {1..$#fList}
        do
            
            # Parameters
            clearNonExistingTrans="y"
            dataModeList="n,n"
            starterModel="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/TrainHistones/Model/blank.hmm"
            fl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/TrainHistones/Annotation/stt/"
            trainSet=$fl$fList[$i]".stt"
            sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/InputSignal/"$model"/"
            hn=$sList[$i]
            signalList=$sl$hn"_norm.bw,"$sl$hn"_slope.bw"
            outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/TrainHistones/Model/"$model"/"
            mkdir -p $outputLocation

            # Execution
            trainHMMEgg $clearNonExistingTrans $dataModeList $starterModel $trainSet $signalList $outputLocation
            mv $outputLocation"model.hmm" $outputLocation$fList[$i]".hmm"

        done
    done
done


