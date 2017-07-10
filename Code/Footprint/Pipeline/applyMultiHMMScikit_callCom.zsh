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
        signalTypeList=( "hd" "local_98" )
        outputTypeList=( "DH_HMM_HD" "DH_HMM_LOC" )
        
        # Grid Loop
        for k in {1..$#signalTypeList}
        do

            # Histone Parameters
            hList=( "H2AZ" "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" )
            fList=( "H2AZ_proximal_com" "H3K4me1_distal_com" "H3K4me3_proximal_com" "H3K9ac_proximal_com" "H3K27ac_proximal_com" )
            oList=( "H2AZ_com" "H3K4me1_com" "H3K4me3_com" "H3K9ac_com" "H3K27ac_com" )

            # Histone Loop
            for i in {1..$#hList}
            do

                # Parameters
                hmmType="4"
                coordFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$sigCell"/InputRegions/hd.bed"
                hmmFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$modCell"/Train/Model/local_98/"$fList[$i]".hmm"
                sigType=$signalTypeList[$k]
                sl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$sigCell"/InputSignal/"$sigType"/"
                hn=$hList[$i]
                signalList=$sl"DNase_norm.bw,"$sl$hn"_norm.bw"
                outType=$outputTypeList[$k]
                ol="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$sigCell"/"$outType"/"
                outputFileName=$ol$oList[$i]"_Model"$modCell".bed"
                mkdir -p $ol

                # Execution
                bsub -J $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_AMHSC" -o $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_AMHSC_out.txt" -e $sigCell"_"$modCell"_"$sigType"_"$oList[$i]"_AMHSC_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyMultiHMMScikit_pipeline.zsh $hmmType $coordFileName $hmmFileName $signalList $outputFileName

            done
        done
    done
done


