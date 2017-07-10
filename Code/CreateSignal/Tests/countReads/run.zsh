#!/bin/zsh

leftInc="0"
rightInc="1"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/K562/DNase.bam"
csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes"
outputFileName="./result.wig"

countReads $leftInc $rightInc $bamFileName $csFileName $outputFileName
