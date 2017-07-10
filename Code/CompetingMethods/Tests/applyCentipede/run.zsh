#!/bin/zsh
mpbsName="CTCF"
mpbsFileName="mpbs.bed"
dampFileName="damp.txt"
priorFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/3.CountReads/Tests/createCentipedeSignal/CTCF_PRIOR.txt"
dnaseFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/3.CountReads/Tests/createCentipedeSignal/CTCF_DNASE.txt"
outputFileName="./CentipedeResults.bed"
applyCentipede $mpbsName $mpbsFileName $dampFileName $priorFileName $dnaseFileName $outputFileName
