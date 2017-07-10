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
            #tfbsList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF"
            #           "EGR1" "FOSL1" "GABP" "JUN" "JUND"
            #           "MAFK" "MAX" "MYC" "NRF1" "P300"
            #           "POU5F1" "RAD21" "REST" "RFX5" "RXRA"
            #           "SIX5" "SP1" "SP2" "SP4" "SRF"
            #           "TBP" "TCF12" "USF1" "USF2" "YY1"
            #           "ZNF143" )
            tfbsList=( "POU5F1" "SP1" )
        else
            #tfbsList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" 
            #           "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" 
            #           "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" 
            #           "EJUND" "ELF1" "ELK1" "ETS1" "FOS" 
            #           "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" 
            #           "JUN" "JUND" "MAFF" "MAFK" "MAX" 
            #           "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" 
            #           "NR2F2" "NRF1" "PU1" "RAD21" "REST" 
            #           "RFX5" "SIX5" "SMC3" "SP1" "SP2" 
            #           "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" 
            #           "TBP" "THAP1" "TR4" "USF1" "USF2" 
            #           "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
            tfbsList=( "IRF1" "ZNF263" )
        fi

        # TF Loop
        for i in {1..$#tfbsList}
        do

            # Parameters
            mpbsName=$tfbsList[$i]
            mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev"/"$mpbsName".bed"
            consFileName="/hpcwork/izkf/projects/egg/Data/PhastCons/PhastCons.bw"
            tssFileName="/hpcwork/izkf/projects/egg/Data/DistanceTSS/distanceTSS.bw"
            dnaseTotExt="200"
            dnasePosFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/DNasePos.bw"
            dnaseNegFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/DNaseNeg.bw"
            outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/Centipede/"

            # Execution
            bsub -J $cell"_"$ev"_"$mpbsName"_CCS" -o $cell"_"$ev"_"$mpbsName"_CCS_out.txt" -e $cell"_"$ev"_"$mpbsName"_CCS_err.txt" -W 100:00 -M 16000 -S 100 -R "select[hpcwork]" -P izkf ./createCentipedeSignal_pipeline.zsh $mpbsName $mpbsFileName $consFileName $tssFileName $dnaseTotExt $dnasePosFileName $dnaseNegFileName $outputLocation

        done
    done
done


