#!/bin/zsh

regionFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyDnase2Tf/regions.bed"
dataFilePath="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createDNase2TfSignal/result/DNase"
mapFilePath="/hpcwork/izkf/projects/TfbsPrediction/Data/GersteinMappability/HG19/"
dtfFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_K562/Dnase2Tf/dinuc_freq_table_ac_DNase.txt"
genomePath="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/"
outputFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyDnase2Tf/result/fp.bed"

applyDnase2Tf $regionFileName $dataFilePath $mapFilePath $dtfFileName $genomePath $outputFileName
