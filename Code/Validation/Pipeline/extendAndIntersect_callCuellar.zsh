#!/bin/zsh

################################################
### CUELLAR PREDICTIONS
################################################

# Cell Line Parameters
cellList=( "H1hesc" "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Predictions Parameters
        predList=( "DNase_priorH3K4me1_priorH3K4me3_prior_"
                   "DNase_priorH3K4me3_priorH3K9ac_prior_"
                   "DNase_priorH3K4me1_priorH3K4me3_priorH3K9ac_prior_" )
        predLabelList=( "D13" "D39" "D139" )

        # Predictions Loop
        for i in {1..$#predList}
        do

            pred=$predList[$i]
            predLabel=$predLabelList[$i]

            # Factor Parameters
            if [[ $cell == "H1hesc" ]]; then
                tfbsList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF" "EGR1" "FOSL1" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NRF1" "P300"
                           "POU5F1" "RAD21" "REST" "RFX5" "RXRA" "SIX5" "SP1" "SP2" "SP4" "SRF" "TBP" "TCF12" "USF1" "USF2" "YY1" "ZNF143" )
            else
                tfbsList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" "E2F6" "EFOS" "EGATA" 
                           "EGR1" "EJUNB" "EJUND" "ELF1" "ELK1" "ETS1" "FOS" "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" 
                           "MAFF" "MAFK" "MAX" "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" 
                           "RFX5" "SIX5" "SMC3" "SP1" "SP2" "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" "TBP" "THAP1" "TR4" "USF1" "USF2" 
                           "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
            fi

            # Factor Loop
            for tfbs in $tfbsList
            do

                # Parameters
                mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev"/"$tfbs".bed"
                predFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/Cuellar5/"$pred$tfbs".bed"
                outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/Cuellar/Cuellar5_"$predLabel"_"$tfbs".bed"

                # Execution
                bsub -J $cell"_"$ev"_"$predLabel"_"$tfbs"_EIC" -o $cell"_"$ev"_"$predLabel"_"$tfbs"_EIC_out.txt" -e $cell"_"$ev"_"$predLabel"_"$tfbs"_EIC_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./extendAndIntersect_pipeline.zsh "0" "0" $mpbsFileName $predFileName $outputFileName

            done
        done
    done
done


