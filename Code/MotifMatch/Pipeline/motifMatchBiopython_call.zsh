#!/bin/zsh

# Parameters
genomeFile="/hpcwork/izkf/projects/egg/Data/HG19/hg19.fa"
searchMethod="biopython"
scoringMethod="fpr"
pseudoC="0.00"
outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/All/fdr_4/"
printBB="N"

# Variation
pwmLoc="/hpcwork/izkf/projects/egg/Data/PWM/"
fprList=( "0.0001" )

#pwmListOld=( "ARID3A_MA0151" "ATF1_UP00020" "ATF3_VATF3Q6" "BACH1_VBACH101" "BHLHE40_UP00050"
#          "BRCA1_MA0133" "CDP_VCDP01" "CEBPB_VCEBPB01" "CEBPD_VCEBPDELTAQ6" "CREB1_MA0018"
#          "CTCF_MA0139" "E2F4_VE2F4DP101" "EGR1_MA0162" "ELF1_VELF1Q6" "ELK1_MA0028"
#          "ETS1_MA0098" "GABP_M00108" "GATA1_M00128" "GATA2_M00348" "GCN4_M00038"
#          "IRF1_MA0050" "JUN_M00036" "MAFK_UP00044" "MAX_M00118" "MAZ_VMAZQ6"
#          "MEF2A_MA0052" "MYC_MA0147" "MYCN_MA0104" "NFE2_M00037" "NFYA_MA0060"
#          "NR2F2_UP00009" "NRF1_VNRF1Q6" "P300_VP30001" "POU5F1_MA0142" "PU1_VPU1Q6"
#          "REST_M00256" "RXRA_UP00053" "SP1_MA0079" "SP4_UP00002" "SRF_MA0083"
#          "STAT1_MA0137" "STAT5A_VSTAT5A01" "TAL1_MA0140" "USF1_MA0093" "USF2_VUSF2Q6"
#          "YY1_M00069" "ZNF143_MA0088" )
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
for fprV in $fprList
do
    for pwm in $pwmList
    do
        pwm_list="-pwm_list="$pwmLoc$pwm".pfm"
        genome_list="-genome_list="$genomeFile
        organism="-organism=hg19"
        search_method="-search_method="$searchMethod
        scoring_method="-scoring_method="$scoringMethod
        pseudocounts="-pseudocounts="$pseudoC
        fpr="-fpr="$fprV
        outLoc=$outputLocation$fprV"/"
        output_location="-output_location="$outLoc
        bigbed="-bigbed="$printBB
        mkdir -p $outLoc
        bsub -J $fprV"_"$pwm"_BF" -o $fprV"_"$pwm"_BF_out.txt" -e $fprV"_"$pwm"_BF_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./motifMatchBiopython_pipeline.zsh $pwm_list $genome_list $organism $search_method $scoring_method $pseudocounts $fpr $output_location $bigbed
    done
done


