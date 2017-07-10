#!/bin/zsh

extParam="4,-5,0,1"
fBiasFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/KmerBias/KmerResults/Tables/HS/K562_8_F.txt"
rBiasFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/KmerBias/KmerResults/Tables/HS/K562_8_R.txt"
coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CreateSignal/Tests/biasSignalATAC/coord.bed"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/K562/ATAC.bam"
fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes"
outputFileName="./signal.wig"

biasSignalATAC $extParam $fBiasFileName $rBiasFileName $coordFileName $bamFileName $fastaFileName $csFileName $outputFileName


