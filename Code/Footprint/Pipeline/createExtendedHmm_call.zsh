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
            fList=( "H3K4me1_distal" "H3K4me1_proximal" "H3K4me3_distal" "H3K4me3_proximal"
                    "H2AZ_proximal" "H3K9ac_proximal" "H3K27ac_proximal" )

            # Signal Loop
            for fileName in $fList
            do
            
                # Parameters
                fl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/"$cell"/Train/Model/"$normType"_"$per"/"

                # Execution
                createExtendedHmm $fl$fileName".hmm" $fl

            done
        done
    done
done


