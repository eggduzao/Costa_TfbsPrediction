#!/bin/zsh

###################################
### Zenke Histones
###################################

# Parameters
#folderList=( "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/cDC_WT_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/cDC_WT_H3K9me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/cDC_WT_H3K27me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/CDP_KO_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/CDP_KO_H3K27me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/CDP_WT_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/CDP_WT_H3K9me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/CDP_WT_H3K27me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/MPP_KO_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/MPP_KO_H3K27me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/MPP_WT_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/MPP_WT_H3K9me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/MPP_WT_H3K27me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/pDC_WT_h3k4me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/pDC_WT_H3K4me3_all/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/pDC_WT_H3K9me3/"
#             "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/pDC_WT_H3K27me3/" )
#chromSizesFile="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/mm9.chrom.sizes"

#folderList=( "/hpcwork/izkf/projects/dendriticcells/data/zenke_histones/macs/pDC_WT_H3K4me3_all/" )
#chromSizesFile="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/mm9.chrom.sizes"

# Execution
#for fn in $folderList
#do
#    lastName=(${(s:/:)fn})
#    lastName=$lastName[-1]
#    bsub -J $lastName"_ZENKE" -o $lastName"_ZENKE_out.txt" -e $lastName"_ZENKE_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./afterMACS_pipeline.zsh $fn $chromSizesFile
#done

###################################
### pDC
###################################

# Parameters
#folderList=( "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.00001_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000001_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000002_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000003_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000004_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000005_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000006_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000007_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000008_5/"
#             "/work/eg474423/ig440396_dendriticcells/local/results/pDC/pDC_H3K4me3_0.000009_5/" )
#chromSizesFile="/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/RegulatoryAnalysisTools/data/mm9/chrom.sizes"

# Execution
#for fn in $folderList
#do
#    lastName=(${(s:/:)fn})
#    lastName=$lastName[-1]
#    bsub -J $lastName"_pDC" -o $lastName"_pDC_out.txt" -e $lastName"_pDC_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./afterMACS_pipeline.zsh $fn $chromSizesFile
#done

###################################
### Stem Aging
###################################

# Parameters
#folderList=( "/work/eg474423/ig440396_dendriticcells/local/stemaging/macs/H3K4me1/"
#             "/work/eg474423/ig440396_dendriticcells/local/stemaging/macs/H3K4me3/"
#             "/work/eg474423/ig440396_dendriticcells/local/stemaging/macs/H3K9me3/"
#             "/work/eg474423/ig440396_dendriticcells/local/stemaging/macs/H3K27me3/"
#             "/work/eg474423/ig440396_dendriticcells/local/stemaging/macs/H3K36me3/" )
#chromSizesFile="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/hg19.chrom.sizes"

# Execution
#for fn in $folderList
#do
#    lastName=(${(s:/:)fn})
#    lastName=$lastName[-1]
#    bsub -J $lastName"_STEM" -o $lastName"_STEM_out.txt" -e $lastName"_STEM_err.txt" -W 10:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./afterMACS_pipeline.zsh $fn $chromSizesFile
#done


###################################
### pDC
###################################

# Parameters
folderList=( "/hpcwork/izkf/projects/stemaging/bedfiles/macs/293Trex_FH.H3K27me3.v2m1hg19f/"
             "/hpcwork/izkf/projects/stemaging/bedfiles/macs/293Trex_FH-CBX2.v2m1hg19f/"
             "/hpcwork/izkf/projects/stemaging/bedfiles/macs/293Trex_FH-RING1B.v2m1hg19f/" )
chromSizesFile="/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/RegulatoryAnalysisTools/data/hg19/chrom.sizes"

# Execution
for fn in $folderList
do
    lastName=(${(s:/:)fn})
    lastName=$lastName[-1]
    bsub -J $lastName"_pDC" -o $lastName"_pDC_out.txt" -e $lastName"_pDC_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./afterMACS_pipeline.zsh $fn $chromSizesFile
done


