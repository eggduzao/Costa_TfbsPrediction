#!/bin/zsh

bedFileName="../CTCF_best.bed"
genomeFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
outputLocation="./"

applyExtreme $bedFileName $genomeFileName $outputLocation


