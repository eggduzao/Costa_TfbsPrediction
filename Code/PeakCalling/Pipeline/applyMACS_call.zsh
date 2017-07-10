#!/bin/zsh

###################################################################################################
##### TFBS
###################################################################################################

# H1hesc
#sigLoc="/hpcwork/izkf/projects/egg/Data/TFBS/H1hesc/"
#outLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/H1hesc/"
#treatmentList=( "HAIB_CTCF_V0416102" "HAIB_GABP_PCR1X" "HAIB_MAX_V0422111" "HAIB_REST_V0416102" "HAIB_SRF_PCR1X"
#                "HAIB_YY1_V0416102" "SYDH_BRCA1_IGGRAB" "SYDH_JUN_IGGRAB" "SYDH_MAFK_IGGRAB" "SYDH_MYC_IGGRAB" 
#                "SYDH_ZNF143_IGGRAB" "HAIB_ATF3_V0416102" "HAIB_CREB1_V0422111" "HAIB_EGR1_V0416102" "HAIB_P300_V0416102"
#                "HAIB_POU5F1_V0416102" "HAIB_RXRA_V0416102" "HAIB_SP1_PCR1X" "HAIB_SP4_V0422111" "HAIB_USF1_PCR1X" 
#                "SYDH_BACH1_IGGRAB" "SYDH_CEBPB_IGGRAB" "SYDH_NRF1_IGGRAB" "SYDH_USF2_IGGRAB" )
#controlList=( "HAIB_CONTROL_V0416102" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_PCR1X"
#              "HAIB_CONTROL_V0416102" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" 
#              "SYDH_CONTROL_IGGRAB" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0416102"
#              "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_PCR1X"
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" )
#nameList=( "CTCF" "GABP" "MAX" "REST" "SRF" 
#           "YY1" "BRCA1" "JUN" "MAFK" "MYC"
#           "ZNF143" "ATF3" "CREB1" "EGR1" "P300"
#           "POU5F1" "RXRA" "SP1" "SP4" "USF1" 
#           "BACH1" "CEBPB" "NRF1" "USF2" )

#treatmentList=( "HAIB_ATF3_V0416102" "HAIB_CREB1_V0422111" "HAIB_EGR1_V0416102" "HAIB_P300_V0416102" "HAIB_POU5F1_V0416102"
#                "HAIB_RXRA_V0416102" "HAIB_SP1_PCR1X" "HAIB_SP4_V0422111" "HAIB_USF1_PCR1X" 
#                "SYDH_BACH1_IGGRAB" "SYDH_CEBPB_IGGRAB" "SYDH_NRF1_IGGRAB" "SYDH_USF2_IGGRAB" )
#controlList=( "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0416102" "HAIB_CONTROL_V0416102"
#              "HAIB_CONTROL_V0416102" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_PCR1X"
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" )
#nameList=( "ATF3" "CREB1" "EGR1" "P300" "POU5F1"
#           "RXRA" "SP1" "SP4" "USF1" 
#           "BACH1" "CEBPB" "NRF1" "USF2" )

# K562
#sigLoc="/hpcwork/izkf/projects/egg/Data/TFBS/K562/"
#outLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/K562/"
#treatmentList=( "HAIB_CTCF_PCR1X" "HAIB_GABP_V0416101" "HAIB_GATA2_PCR1X" "HAIB_MAX_V0416102" "HAIB_REST_V0416102"
#                "HAIB_SRF_V0416101" "HAIB_YY1_V0416101" "SYDH_ARID3A_IGGRAB" "SYDH_ATF1_STD" "SYDH_BHLHE40_IGGRAB" 
#                "SYDH_ELK1_IGGRAB" "SYDH_FOS_STD" "SYDH_GATA1_UCD" "SYDH_IRF1_IFNA6H" "SYDH_JUN_STD"
#                "SYDH_MAFK_IGGRAB" "SYDH_MYC_STD" "SYDH_NFE2_STD" "SYDH_NFYA_STD" "SYDH_STAT1_IFNA6H"
#                "SYDH_TAL1_IGGMUS" "SYDH_YY1_UCD" "SYDH_ZNF143_IGGRAB" "HAIB_ATF3_V0416101" "HAIB_CEBPB_V0422111"
#                "HAIB_CEBPD_V0422111" "HAIB_CREB1_V0422111" "HAIB_EGR1_V0416101" "HAIB_ELF1_V0416102" "HAIB_ETS1_V0416101"
#                "HAIB_MEF2A_V0416101" "HAIB_NR2F2_V0422111" "HAIB_PU1_PCR1X" "HAIB_SP1_PCR1X" "HAIB_STAT5A_V0422111"
#                "HAIB_USF1_V0416101" "SYDH_BACH1_IGGRAB" "SYDH_CDP_IGGRAB" "SYDH_E2F4_UCD" "SYDH_MAZ_IGGRAB"
#                "SYDH_NRF1_IGGRAB" "SYDH_P300_IGGRAB" "SYDH_USF2_IGGRAB" )
#controlList=( "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101"
#              "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_STD" "SYDH_CONTROL_IGGRAB" 
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_STD" "SYDH_CONTROL_UCD" "SYDH_CONTROL_IFNA6H" "SYDH_CONTROL_STD" 
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_STD" "SYDH_CONTROL_STD" "SYDH_CONTROL_STD" "SYDH_CONTROL_IFNA6H" 
#              "SYDH_CONTROL_IGGMUS" "SYDH_CONTROL_UCD" "SYDH_CONTROL_IGGRAB" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0422111"
#              "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101"
#              "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0422111"
#              "HAIB_CONTROL_V0416101" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_UCD" "SYDH_CONTROL_IGGRAB" 
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" )
#nameList=( "CTCF" "GABP" "GATA2" "MAX" "REST"
#           "SRF" "YY1" "ARID3A" "ATF1" "BHLHE40" 
#           "ELK1" "FOS" "GATA1" "IRF1" "JUN"
#           "MAFK" "MYC" "NFE2" "NFYA" "STAT1"
#           "TAL1" "YY1" "ZNF143" "ATF3" "CEBPB"
#           "CEBPD" "CREB1" "EGR1" "ELF1" "ETS1"
#           "MEF2A" "NR2F2" "PU1" "SP1" "STAT5A"
#           "USF1" "BACH1" "CDP" "E2F4" "MAZ"
#           "NRF1" "P300" "USF2" )

#treatmentList=( "HAIB_ATF3_V0416101" "HAIB_CEBPB_V0422111" "HAIB_CEBPD_V0422111" "HAIB_CREB1_V0422111" "HAIB_EGR1_V0416101"
#                "HAIB_ELF1_V0416102" "HAIB_ETS1_V0416101" "HAIB_MEF2A_V0416101" "HAIB_NR2F2_V0422111" "HAIB_PU1_PCR1X"
#                "HAIB_SP1_PCR1X" "HAIB_STAT5A_V0422111" "HAIB_USF1_V0416101"
#                "SYDH_BACH1_IGGRAB" "SYDH_CDP_IGGRAB" "SYDH_E2F4_UCD" "SYDH_MAZ_IGGRAB" "SYDH_NRF1_IGGRAB"
#                "SYDH_P300_IGGRAB" "SYDH_USF2_IGGRAB" )
#controlList=( "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416101"
#              "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0416101" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_PCR1X"
#              "HAIB_CONTROL_PCR1X" "HAIB_CONTROL_V0422111" "HAIB_CONTROL_V0416101"
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_UCD" "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB"
#              "SYDH_CONTROL_IGGRAB" "SYDH_CONTROL_IGGRAB" )
#nameList=( "ATF3" "CEBPB" "CEBPD" "CREB1" "EGR1"
#           "ELF1" "ETS1" "MEF2A" "NR2F2" "PU1"
#           "SP1" "STAT5A" "USF1"
#           "BACH1" "CDP" "E2F4" "MAZ" "NRF1"
#           "P300" "USF2" )

###################################################################################################
##### HISTONES
###################################################################################################

# H1hesc
#sigLoc="/hpcwork/izkf/projects/egg/Data/Histone/H1hesc/"
#outLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HistonePeaks/H1hesc/"
#treatmentList=( "H2AZ" "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" )
#controlList=( "Control" "Control" "Control" "Control" "Control" )
#nameList=( "H2AZ" "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" )

# K562
#sigLoc="/hpcwork/izkf/projects/egg/Data/Histone/K562/"
#outLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HistonePeaks/K562/"
#treatmentList=( "H2AZ" "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" )
#controlList=( "Control" "Control" "Control" "Control" "Control" )
#nameList=( "H2AZ" "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" )

# Huvec
sigLoc="/hpcwork/izkf/projects/egg/Data/Histone/Huvec/"
outLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HistonePeaks/Huvec/"
treatmentList=( "H3K4me1" "H3K4me3" )
controlList=( "Control" "Control" )
nameList=( "H3K4me1" "H3K4me3" )

###################################################################################################
##### Execution Loop
###################################################################################################

for i in {1..$#nameList}
do
    outputLocation=$outLoc$nameList[$i]"/"
    mkdir -p $outputLocation
    bsub -J $nameList[$i]"_PCK" -o $nameList[$i]"_PCK_out.txt" -e $nameList[$i]"_PCK_err.txt" -W 100:00 -M 12000 -S 100 -R "select[hpcwork]" ./applyMACS_pipeline.zsh $nameList[$i] $sigLoc$treatmentList[$i]".bam" $sigLoc$controlList[$i]".bam" $outputLocation
done

# -P izkf

