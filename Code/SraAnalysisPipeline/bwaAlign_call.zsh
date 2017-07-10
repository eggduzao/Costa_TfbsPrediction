#!/bin/zsh

###########################
# NakedK562
###########################

# Input
genomeFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
readsFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedK562/SRR1565781.fastq"
outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedK562_2/NakedK562"

# Running bwa
bsub -J "BWA_NK5" -o "BWA_NK5_out.txt" -e "BWA_NK5_err.txt" -W 100:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./bwaAlign_pipeline.zsh  $genomeFileName $readsFileName $outputFileName

###########################
# NakedMcf7
###########################

# Input
genomeFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
readsFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedMcf7/SRR1565782.fastq"
outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedMcf7_2/NakedMcf7"

# Running bwa
bsub -J "BWA_NMC" -o "BWA_NMC_out.txt" -e "BWA_NMC_err.txt" -W 100:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./bwaAlign_pipeline.zsh  $genomeFileName $readsFileName $outputFileName


