#!/bin/zsh

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "LnCaP" "m3134" "Mcf7" "Saos2" "denovo" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    #evList=( "fdr_4" "bitscore_log" )
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # TF parameters
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
        elif [[ $cell == "HeLaS3" ]]; then
            mpbsList=( "BRCA1_MA0133_1" "CEBPB_MA0466_1" "CTCF_MA0139_1" "E2F4_MA0470_1" "E2F6_MA0471_1"
                       "ELK1_MA0028_1" "FOS_MA0476_1" "GABP_MA0062_2" "JUN_MA0488_1" "JUND_MA0491_1"
                       "MAFK_MA0496_1" "MAX_MA0058_2" "MYC_MA0147_2" "NFYA_MA0060_2" "NFYB_MA0502_1"
                       "NRF1_MA0506_1" "P300_M00033" "CTCF_MA0139_1" "REST_MA0138_2" "STAT1_MA0137_3"
                       "TBP_MA0108_2" "USF2_MA0526_1" "ZNF143_MA0088_1" )
            tfbsList=( "BRCA1" "CEBPB" "CTCF" "E2F4" "E2F6"
                       "ELK1" "FOS" "GABP" "JUN" "JUND"
                       "MAFK" "MAX" "MYC" "NFYA" "NFYB"
                       "NRF1" "P300" "RAD21" "REST" "STAT1"
                       "TBP" "USF2" "ZNF143" )
        elif [[ $cell == "HepG2" ]]; then
            mpbsList=( "ARID3A_MA0151_1" "CREB1_MA0018_2" "BHLHE40_MA0464_1" "BRCA1_MA0133_1" "CEBPB_MA0466_1"
                       "CTCF_MA0139_1" "ELF1_MA0473_1" "GABP_MA0062_2" "JUN_MA0488_1" "JUND_MA0491_1"
                       "MAFF_MA0495_1" "MAFK_MA0496_1" "MAX_MA0058_2" "MYC_MA0147_2" "NRF1_MA0506_1"
                       "P300_M00033" "CTCF_MA0139_1" "REST_MA0138_2" "RXRA_MA0512_1" "SP1_MA0079_3"
                       "SP2_MA0516_1" "SRF_MA0083_2" "TBP_MA0108_2" "USF1_MA0093_2" "USF2_MA0526_1"
                       "YY1_MA0095_2" )
            tfbsList=( "ARID3A" "ATF3" "BHLHE40" "BRCA1" "CEBPB"
                       "CTCF" "ELF1" "GABP" "JUN" "JUND"
                       "MAFF" "MAFK" "MAX" "MYC" "NRF1"
                       "P300" "RAD21" "REST" "RXRA" "SP1"
                       "SP2" "SRF" "TBP" "USF1" "USF2"
                       "YY1" )
        elif [[ $cell == "Huvec" ]]; then
            mpbsList=( "CTCF_MA0139_1" "FOS_MA0476_1" "GATA2_MA0036_2" "JUN_MA0488_1" "MAX_MA0058_2"
                       "MYC_MA0147_2" )
            tfbsList=( "CTCF" "FOS" "GATA2" "JUN" "MAX"
                       "MYC" )
        elif [[ $cell == "K562" ]]; then
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
        elif [[ $cell == "LnCaP" ]]; then
            mpbsList=( "AR" "AR" "AR" "AR" )
            tfbsList=( "AR_1nmR" "AR_10nmR" "AR_ethl" "AR_R1881" )
        elif [[ $cell == "m3134" ]]; then
            mpbsList=( "GR" "GR" )
            tfbsList=( "GR_withDEX" "GR_woDEX" )
        elif [[ $cell == "Mcf7" ]]; then
            mpbsList=( "ER" "ER" "ER" "ER" "ER" "ER" )
            tfbsList=( "ER_0min" "ER_2min" "ER_5min" "ER_10min" "ER_40min" "ER_160min" )
        elif [[ $cell == "Saos2" ]]; then
            mpbsList=( "P53" )
            tfbsList=( "P53" )
        elif [[ $cell == "denovo" ]]; then
            mpbsList=( "UW_Motif_0458" "UW_Motif_0500" )
            tfbsList=( "UW_Motif_0458" "UW_Motif_0500" )
        fi

        # TF Loop
        for i in {1..$#mpbsList}
        do

            # Parameters
            peakExt="100"
            mpbsName=$mpbsList[$i]
            tfbsName=$tfbsList[$i]
            tfbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/TFBSAWG/"$cell"/"$tfbsName".bed"
            mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/"$ev"/"$mpbsName".bed"
            outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev"/"
            mkdir -p $outputLocation

            # Execution
            #bsub -J $cell"_"$ev"_"$tfbsName"_CE" -o $cell"_"$ev"_"$tfbsName"_CE_out.txt" -e $cell"_"$ev"_"$tfbsName"_CE_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./createEvidence_pipeline.zsh $peakExt $mpbsName $tfbsFileName $mpbsFileName $outputLocation
            ./createEvidence_pipeline.zsh $peakExt $mpbsName $tfbsFileName $mpbsFileName $outputLocation

        done
    done
done


