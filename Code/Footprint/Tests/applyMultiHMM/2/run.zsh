#!/bin/zsh

proximalFile="/hpcwork/izkf/projects/egg/Data/DistanceTSS/distanceTSS.bw"
dnaseSig="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/DNase.bw"
histSig="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/H3K4me3.bw"

applyMultiHMM mpbs.bed $proximalFile "HMM/dist.hmm" "HMM/prox.hmm" $dnaseSig","$histSig "./out/"


