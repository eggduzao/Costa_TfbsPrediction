#!/bin/zsh

fBiasFileName="/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_K562_6_F.txt"
rBiasFileName="/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/DNase_K562_6_R.txt"
coordFileName="/home/egg/Projects/TfbsPrediction/Code/CreateSignal/Tests/biasSignal/coord.bed"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/K562/DNase.bam"
fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes"
outputFileName="./signal.wig"

biasSignal $fBiasFileName $rBiasFileName $coordFileName $bamFileName $fastaFileName $csFileName $outputFileName


