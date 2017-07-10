#!/bin/zsh

cellLineList="H1hesc,HeLaS3,HepG2,Huvec,K562"
fdrList="0.0001,0.00001"
evNameList="fdr_4,bitscore_log"
pwmLocation="/hpcwork/izkf/projects/egg/Data/PWM/Jaspar_Transfac/"
tfbsLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBSAWG/"
mpbsLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"
outputFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TableStatistics/TFBS/TfbsStatistics.tex"
totalExt="100"

bsub -J "TFBS_TAB" -o "TFBS_TAB_out.txt" -e "TFBS_TAB_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./tfbsStatistics_pipeline.zsh $cellLineList $fdrList $evNameList $pwmLocation $tfbsLocation $mpbsLocation $outputFileName $totalExt


