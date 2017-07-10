#!/bin/zsh

# Parameters
il1="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/K562_Evidence/"
il2="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/K562/"
il3="/hpcwork/izkf/projects/TfbsPrediction/Results/SignalProcessing/Results/"
outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/SignalProcessing/ROC/"
mkdir -p $outputLocation

labelList="PWM,TC_DU,FS_DU,butterlow,elliphigh,endstd_1,order_1.2,Order_1,Order_4,Order_8,Startstd_4,Startstd_5,window_20,window_30,wp_0.2,wp_0.6,wp_0.8"
typeList="SC,SC,SC,SB,SB,SB,SB,SB,SB,SB,SB,SB,SB,SB,SB,SB,SB"
bedList=$il1"fdr_4.bed,"$il2"TC_DU.bed,"$il2"FS_DU.bed,"$il3"butterlow.bed,"$il3"elliphigh.bed,"$il3"endstd_1.bed,"$il3"order_1.2.bed,"$il3"Order_1.bed,"$il3"Order_4.bed,"$il3"Order_8.bed,"$il3"Startstd_4.bed,"$il3"Startstd_5.bed,"$il3"window_20.bed,"$il3"window_30.bed,"$il3"wp_0.2.bed,"$il3"wp_0.6.bed,"$il3"wp_0.8.bed"

factorList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" "EJUND" "ELF1" "ELK1" "ETS1" "FOS" "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" "MAFF" "MAFK" "MAX" "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" "RFX5" "SIX5" "SMC3" "SP1" "SP2" "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" "TBP" "THAP1" "TR4" "USF1" "USF2" "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )

# Factor Loop
for factor in $factorList
do
  bsub -J "AHM_ROCS" -o "AHM_ROCS_out.txt" -e "AHM_ROCS_err.txt" -W 48:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./rocFromBedSBSC_pipeline.zsh $factor $typeList $labelList $bedList $outputLocation
done


