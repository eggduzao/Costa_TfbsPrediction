#!/bin/zsh

mpbsName="CTCF"
typeList="SB,SB,SC,SC"
labelList="BOYLE,BIAS,PIQUE,PWM"
bedList="./resultsBoyle.bed,./resultsBIAS.bed,./resultsPiquet.bed,./resultsPwm.bed"
outputLocation="./"

rocFromBedSBSC $mpbsName $typeList $labelList $bedList $outputLocation


