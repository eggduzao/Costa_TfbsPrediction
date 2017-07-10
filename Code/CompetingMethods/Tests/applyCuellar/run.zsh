#!/bin/zsh
# Para rodar esse teste, favor adicionar no applyCuellar.py:
# chromSizesFile = open("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyCuellar/priorLocation/chrom.sizes","r") ####### TODO ERASE
# chrList = ["chr1", "chr2"] ############ TODO ERASE
mpbsName="GABP"
fimoThresh="0.0001"
memeFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/MEME/GABP_MA0062_2.meme"
priorFileNameList="./priorLocation/prior1.bw"
genomeLocation="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/"
outputLocation="./out/"
tempLocation="./temp/"
mkdir -p $tempLocation
applyCuellar $mpbsName $fimoThresh $memeFileName $priorFileNameList $genomeLocation $outputLocation $tempLocation
rm -rf $tempLocation
#priorFileNameList="./priorLocation/prior1.bw,./priorLocation/prior2.bw"
#applyCuellar $mpbsName $fimoThresh $memeFileName $priorFileNameList $genomeLocation $outputLocation


