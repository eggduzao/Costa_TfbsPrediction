#!/bin/zsh

# Parameters
genomeFile="/hpcwork/izkf/projects/egg/Data/HG19/hg19.fa"
searchMethod="biopython"
scoringMethod="bitscore"
pseudoC="0.00"
outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/All/bitscore_log/"
printBB="N"

# Variation
pwmLoc="/hpcwork/izkf/projects/egg/Data/PWM/Jaspar/"
scoreList=( "13.2877" )

pwmList=( "ARID3A_MA0151_1" "ATF1_UP00020_1" "BACH1_M00495" "BHLHE40_MA0464_1" "BRCA1_MA0133_1"
          "CEBPB_MA0466_1" "CREB1_MA0018_2" "CTCF_MA0139_1" "E2F4_MA0470_1" "E2F6_MA0471_1"
          "EGR1_MA0162_2" "ELF1_MA0473_1" "ELK1_MA0028_1" "ETS1_MA0098_2" "FOSL1_MA0477_1"
          "FOS_MA0476_1" "GABP_MA0062_2" "GATA1_MA0035_3" "GATA2_MA0036_2" "IRF1_MA0050_2"
          "JUNB_MA0490_1" "JUND_MA0491_1" "JUN_MA0488_1" "MAFF_MA0495_1" "MAFK_MA0496_1"
          "MAX_MA0058_2" "MEF2A_MA0052_2" "MYC_MA0147_2" "NFE2_MA0501_1" "NFYA_MA0060_2"
          "NFYB_MA0502_1" "NR2F2_UP00009_1" "NRF1_MA0506_1" "P300_M00033" "POU5F1_MA0142_1"
          "PU1_MA0080_3" "REST_MA0138_2" "RFX5_MA0510_1" "RXRA_MA0512_1" "SP1_MA0079_3"
          "SP2_MA0516_1" "SP4_UP00002_1" "SRF_MA0083_2" "STAT1_MA0137_3" "STAT2_MA0517_1"
          "STAT5A_MA0519_1" "TAL1_MA0140_2" "TBP_MA0108_2" "TCF12_MA0521_1" "THAP1_MA0597_1"
          "TR4_MA0504_1" "USF1_MA0093_2" "USF2_MA0526_1" "YY1_MA0095_2" "ZBTB7A_UP00047_1"
          "ZBTB33_MA0527_1" "ZNF143_MA0088_1" "ZNF263_MA0528_1" )

# Execution
for score in $scoreList
do
    for pwm in $pwmList
    do
        pwm_list="-pwm_list="$pwmLoc$pwm".pfm"
        genome_list="-genome_list="$genomeFile
        organism="-organism=hg19"
        search_method="-search_method="$searchMethod
        scoring_method="-scoring_method="$scoringMethod
        pseudocounts="-pseudocounts="$pseudoC
        bitscore="-bitscore="$score
        output_location="-output_location="$outputLocation
        bigbed="-bigbed="$printBB
        mkdir -p $outputLocation
        bsub -J $pwm"_MMP" -o $pwm"_MMP_out.txt" -e $pwm"_MMP_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./motifMatchPique_pipeline.zsh $pwm_list $genome_list $organism $search_method $scoring_method $pseudocounts $bitscore $output_location $bigbed
    done
done


