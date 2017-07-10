#!/bin/zsh
mpbsFileName="mpbs.bed"
signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/K562/DNase.bw"
outputFileName="./result.bed"
applyFOS $mpbsFileName $signalFileName $outputFileName
