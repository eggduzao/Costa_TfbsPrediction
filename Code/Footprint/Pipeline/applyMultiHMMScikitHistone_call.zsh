#!/bin/zsh

# Signal Parameters
signalCellList=( "H1hesc" "K562" )

# Signal Loop
for sigCell in $signalCellList
do

    # Model Parameters
    modelCellList=( "H1hesc" "K562" )

    # Model Loop
    for modCell in $modelCellList
    do
    
        # Signal Parameters
        signalTypeList=( "hd" )
        outputTypeList=( "H_HMM" )
        
        # Grid Loop
        for k in {1..$#signalTypeList}
        do

            # Histone Parameters
            hList=( "H3K9ac" )
            fList=( "H3K9ac_proximal" )
            oList=( "H3K9ac" )

            # Histone Loop
            for i in {1..$#hList}
            do

                # Parameters
                coordFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$sigCell"/InputRegions/hd.bed"
                hmmFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$modCell"/TrainHistones/Model/hd/"$fList[$i]".hmm"
                sigType=$signalTypeList[$k]
                sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$sigCell"/InputSignal/"$sigType"/"
                hn=$hList[$i]
                signalList=$sl$hn"_norm.bw,"$sl$hn"_slope.bw"
                outType=$outputTypeList[$k]
                ol="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/"$outType"/"
                outputFileName=$ol$oList[$i]"_Model"$modCell".bed"
                mkdir -p $ol



                # Execution
                bsub -J $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_HHMM" -o $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_HHMM_out.txt" -e $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_HHMM_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyMultiHMMScikitHistone_pipeline.zsh $coordFileName $hmmFileName $signalList $outputFileName

            done
        done
    done
done


