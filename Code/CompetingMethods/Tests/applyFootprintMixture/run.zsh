#!/bin/zsh

mpbsFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/mpbsFileName.bed"
peakFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyFootprintMixture/DNasePeaks.bed"
cutsFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/result/mpbs_CUT.txt"
seqFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/createFootprintMixtureSignal/result/mpbs_SEQ.fa"
outputFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyFootprintMixture/fp.bed"

applyFootprintMixture $mpbsFileName $peakFileName $cutsFileName $seqFileName $outputFileName


