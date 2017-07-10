
# Import
import sys
from Bio import motifs
from glob import glob

# Reading input
inputLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/PearsonNaked/"
inList = [ "DU_H1hesc", "DU_HeLaS3", "DU_HepG2", "DU_Huvec", "DU_K562", "UW_HepG2", "UW_Huvec", "UW_K562", "DU_Mcf7", "DU_Saos2", "UW_denovo", "UW_LnCaP", "UW_m3134_with_DEX", "UW_m3134_wo_DEX" ]
inList = [ "UW_H7hesc" ]

# Execution
for inName in inList:
  for inFileName in glob(inputLoc+inName+"/PWM/*_align.pwm"):
    inFile = open(inFileName,"r")
    outFileName = inFileName[:-3]+"eps"
    pwm = motifs.read(inFile, "pfm")
    #pwm.weblogo(outFileName, format="png_print", stack_width = "medium", stacks_per_line = "50", color_scheme = "color_classic")
    pwm.weblogo(outFileName, format="eps", stack_width = "medium", stacks_per_line = "50", color_scheme = "color_classic")
    inFile.close()


