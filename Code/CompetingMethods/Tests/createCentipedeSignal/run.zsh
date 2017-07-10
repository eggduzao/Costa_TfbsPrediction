#!/bin/zsh
mpbsName="CTCF"
mpbsFileName="mpbs.bed"
consFileName="cons.bw"
tssFileName="tss.bw"
dnaseTotExt="4"
dnasePosFileName="DNasePos.bw"
dnaseNegFileName="DNaseNeg.bw"
outputLocation="./"
createCentipedeSignal $mpbsName $mpbsFileName $consFileName $tssFileName $dnaseTotExt $dnasePosFileName $dnaseNegFileName $outputLocation
