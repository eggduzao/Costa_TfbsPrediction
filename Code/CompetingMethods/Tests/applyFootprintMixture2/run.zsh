#!/bin/zsh

mpbsFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/mpbsFileName.bed"
peakFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyFootprintMixture2/DNasePeaks.bed"
cutsFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/result/mpbs_CUT.txt"
seqFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/result/mpbs_SEQ.fa"
outputFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyFootprintMixture2/fp.bed"

applyFootprintMixture2 $mpbsFileName $peakFileName $cutsFileName $seqFileName $outputFileName


