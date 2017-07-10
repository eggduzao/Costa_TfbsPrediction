#!/bin/zsh

# Threshold Parameters
#fimoThreshList=( "0.0000001" "0.000001" "0.00001" "0.0001" "0.001" )
fimoThreshList=( "0.00001" )

# Threshold Loop
for fimoThresh in $fimoThreshList
do

    # Folder Name
    if [[ $fimoThresh == "0.0000001" ]]; then
        outFoldName="Cuellar7"
    elif [[ $fimoThresh == "0.000001" ]]; then
        outFoldName="Cuellar6"
    elif [[ $fimoThresh == "0.00001" ]]; then
        outFoldName="Cuellar5"
    elif [[ $fimoThresh == "0.0001" ]]; then
        outFoldName="Cuellar4"
    else
        outFoldName="Cuellar3"
    fi
    
    # Cell Line Parameters
    cellList=( "H1hesc" "K562" )

    # Cell Line Loop
    for cell in $cellList
    do

        # Prior Parameters
        pl="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/"$cell"/Cuellar/"
        priorList=( $pl"DNase_prior.bw,"$pl"H3K4me1_prior.bw,"$pl"H3K4me3_prior.bw,"$pl"H3K9ac_prior.bw"
                    $pl"DNase_prior.bw,"$pl"H3K4me3_prior.bw,"$pl"H3K9ac_prior.bw" )
        priorLabelList=( "D139" "D39" )

        # Prior Loop
        for i in {1..$#priorList}
        do

            # Factor Parameters
            if [[ $cell == "H1hesc" ]]; then
                mpbsList=( "USF1_MA0093_2" "BACH1_M00495" "BRCA1_MA0133_1" "CEBPB_MA0466_1" "CTCF_MA0139_1" 
                           "EGR1_MA0162_2" "FOSL1_MA0477_1" "GABP_MA0062_2" "JUN_MA0488_1" "JUND_MA0491_1" 
                           "MAFK_MA0496_1" "MAX_MA0058_2" "MYC_MA0147_2" "NRF1_MA0506_1" "P300_M00033" 
                           "POU5F1_MA0142_1" "CTCF_MA0139_1" "REST_MA0138_2" "RFX5_MA0510_1" "RXRA_MA0512_1" 
                           "ZNF143_MA0088_1" "SP1_MA0079_3" "SP2_MA0516_1" "SP4_UP00002_1" "SRF_MA0083_2" 
                           "TBP_MA0108_2" "TCF12_MA0521_1" "USF1_MA0093_2" "USF2_MA0526_1" "YY1_MA0095_2"
                           "ZNF143_MA0088_1" )
                tfbsList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF"
                           "EGR1" "FOSL1" "GABP" "JUN" "JUND"
                           "MAFK" "MAX" "MYC" "NRF1" "P300"
                           "POU5F1" "RAD21" "REST" "RFX5" "RXRA"
                           "SIX5" "SP1" "SP2" "SP4" "SRF"
                           "TBP" "TCF12" "USF1" "USF2" "YY1"
                           "ZNF143" )
            else
                mpbsList=( "ARID3A_MA0151_1" "ATF1_UP00020_1" "CREB1_MA0018_2" "BACH1_M00495" "BHLHE40_MA0464_1"
                           "TAL1_MA0140_2" "CEBPB_MA0466_1" "CTCF_MA0139_1" "CTCF_MA0139_1" "E2F4_MA0470_1"
                           "E2F6_MA0471_1" "FOS_MA0476_1" "GATA2_MA0036_2" "EGR1_MA0162_2" "JUNB_MA0490_1"
                           "JUND_MA0491_1" "ELF1_MA0473_1" "ELK1_MA0028_1" "ETS1_MA0098_2" "FOS_MA0476_1"
                           "FOSL1_MA0477_1" "GABP_MA0062_2" "GATA1_MA0035_3" "GATA2_MA0036_2" "IRF1_MA0050_2"
                           "JUN_MA0488_1" "JUND_MA0491_1" "MAFF_MA0495_1" "MAFK_MA0496_1" "MAX_MA0058_2"
                           "MEF2A_MA0052_2" "MYC_MA0147_2" "NFE2_MA0501_1" "NFYA_MA0060_2" "NFYB_MA0502_1"
                           "NR2F2_UP00009_1" "NRF1_MA0506_1" "PU1_MA0080_3" "CTCF_MA0139_1" "REST_MA0138_2"
                           "RFX5_MA0510_1" "ZNF143_MA0088_1" "CTCF_MA0139_1" "SP1_MA0079_3" "SP2_MA0516_1"
                           "SRF_MA0083_2" "STAT1_MA0137_3" "STAT2_MA0517_1" "STAT5A_MA0519_1" "TAL1_MA0140_2"
                           "TBP_MA0108_2" "THAP1_MA0597_1" "TR4_MA0504_1" "USF1_MA0093_2" "USF2_MA0526_1"
                           "YY1_MA0095_2" "ZBTB7A_UP00047_1" "ZBTB33_MA0527_1" "ZNF143_MA0088_1" "ZNF263_MA0528_1" )
                tfbsList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" 
                           "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" 
                           "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" 
                           "EJUND" "ELF1" "ELK1" "ETS1" "FOS" 
                           "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" 
                           "JUN" "JUND" "MAFF" "MAFK" "MAX" 
                           "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" 
                           "NR2F2" "NRF1" "PU1" "RAD21" "REST" 
                           "RFX5" "SIX5" "SMC3" "SP1" "SP2" 
                           "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" 
                           "TBP" "THAP1" "TR4" "USF1" "USF2" 
                           "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
            fi

            # Factor Loop
            for j in {1..$#mpbsList}
            do

                # Parameters
                mpbsName=$mpbsList[$j]
                tfbsName=$tfbsList[$j]
                memeFileName="/hpcwork/izkf/projects/egg/Data/PWM/MEME/"$mpbsName".meme"
                plb=$priorLabelList[$i]
                priorFileNameList=$priorList[$i]
                genomeLocation="/hpcwork/izkf/projects/egg/Data/HG19/"
                outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"$outFoldName"/"
                mkdir -p $outputLocation

                # Execution
                bsub -J $outFoldName"_"$cell"_"$plb"_"$tfbsName"_ACP" -o $outFoldName"_"$cell"_"$plb"_"$tfbsName"_ACP_out.txt" -e $outFoldName"_"$cell"_"$plb"_"$tfbsName"_ACP_err.txt" -W 150:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./applyCuellar_pipeline.zsh $tfbsName $fimoThresh $memeFileName $priorFileNameList $genomeLocation $outputLocation

            done
        done
    done
done


