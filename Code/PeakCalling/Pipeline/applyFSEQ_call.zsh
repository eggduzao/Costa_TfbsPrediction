#!/bin/zsh

# Parameters
featLen="300"
bff="/hpcwork/izkf/projects/egg/Data/Epigenetics/EncodeHG19/Background/bff/bff_35/"
iff="/hpcwork/izkf/projects/egg/Data/Epigenetics/EncodeHG19/Background/iff/iff_K562/"
tl="/hpcwork/izkf/projects/egg/Data/Epigenetics/EncodeHG19/TFBS/"
outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/FSEQ/K562/All/"

# Variations
nameList=("CTCF" "GABP" "SRF" "REST" "GATA2" "MAX" "YY1" "GCN4" "JUN" "NFE2" "E2F4" "GATA1")
tfbsList=($tl"HAIB/K562/CTCF/PCR1X/" $tl"HAIB/K562/GABP/V0416101/" $tl"HAIB/K562/SRF/V0416101/" $tl"HAIB/K562/REST/V0416102/" $tl"HAIB/K562/GATA2/PCR1X/" $tl"HAIB/K562/MAX/V0416102/" $tl"HAIB/K562/YY1/V0416101/" $tl"SYDH/K562/GCN4/STD/" $tl"SYDH/K562/JUN/STD/" $tl"SYDH/K562/NFE2/STD/" $tl"SYDH/K562/E2F4/UCD/" $tl"SYDH/K562/GATA1/UCD/")

# Execution
for i in {1..$#tfbsList}
do
    mkdir -p $outputLocation$nameList[$i]"/"
    bsub -J $nameList[$i]"_FSEQ" -o $nameList[$i]"_FSEQ_out.txt" -e $nameList[$i]"_FSEQ_err.txt" -W 100:00 -M 12000 -S 100 -P izkf ./applyFSEQ_pipeline.zsh $featLen $bff $iff $tfbsList[$i] $outputLocation$nameList[$i]"/"
done


