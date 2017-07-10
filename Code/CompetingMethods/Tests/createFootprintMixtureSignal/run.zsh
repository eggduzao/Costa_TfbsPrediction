#!/bin/zsh
mpbsFileName="./mpbsFileName.bed"
bamFileName="./Signal.bam"
genomeFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
outputFileName="./result/mpbs"
createFootprintMixtureSignal $mpbsFileName $bamFileName $genomeFileName $outputFileName
