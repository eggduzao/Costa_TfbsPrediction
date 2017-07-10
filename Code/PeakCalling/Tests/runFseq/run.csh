#!/bin/csh

# Server
#set backLoc = "/project1/Epigenetics/data/encode/hg19/Background/bff/"
#set ploidyLoc = "/project1/Epigenetics/data/encode/hg19/Background/iff/"
#set outputLoc = "/home/egg/GeneRegulation/exp/3.CountReads/Tests/runFseq/"

# Local
set backLoc = "/home/egg/Project/Data/Background/bff/"
set ploidyLoc = "/home/egg/Project/Data/Background/iff/"
set outputLoc = "/home/egg/Project/GeneRegulation/exp/3.CountReads/Tests/runFseq/"

runFseq 0 600 $backLoc $ploidyLoc $outputLoc $outputLoc"cell1/" $outputLoc"cell2/" $outputLoc"cell3/"
