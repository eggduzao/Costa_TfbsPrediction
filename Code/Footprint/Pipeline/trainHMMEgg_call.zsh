#!/bin/zsh

# Cell Line Parameters
#cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
cellList=( "HeLaS3" "HepG2" "Huvec" )

# Cell Loop
for cell in $cellList
do

    # Model Parameters
    modelList=( "hd" )

    # Grid Loop
    for model in $modelList
    do
    
        # Signal Parameters
        #sList=( "H3K4me1" "H3K4me1" "H3K4me3" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
        #fList=( "H3K4me1_distal" "H3K4me1_proximal" "H3K4me3_distal" "H3K4me3_proximal" "H2AZ_proximal" "H3K9ac_proximal" "H3K27ac_proximal" )
        sList=( "H3K4me1" "H3K4me3" )
        fList=( "H3K4me1_distal" "H3K4me3_proximal" )

        # Signal Loop
        for i in {1..$#fList}
        do
            
            # Parameters
            clearNonExistingTrans="y"
            dataModeList="n,n,n,n"
            starterModel="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/blank.hmm"
            fl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Annotation/stt/"
            trainSet=$fl$fList[$i]".stt"
            sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/InputSignal/"$model"/"
            hn=$sList[$i]
            signalList=$sl"DNase_norm.bw,"$sl"DNase_slope.bw,"$sl$hn"_norm.bw,"$sl$hn"_slope.bw"
            outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/"$model"/"
            mkdir -p $outputLocation

            # Execution
            trainHMMEgg $clearNonExistingTrans $dataModeList $starterModel $trainSet $signalList $outputLocation
            mv $outputLocation"model.hmm" $outputLocation$fList[$i]".hmm"

        done
    done
done


