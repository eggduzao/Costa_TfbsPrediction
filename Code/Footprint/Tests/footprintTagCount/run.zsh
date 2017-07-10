#!/bin/zsh

windowLen="20"
fpFileName="mpbs.bed"
signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_K562/DNase.bw"
outputFileName="./result.bed"

footprintTagCount $windowLen $fpFileName $signalFileName $outputFileName


