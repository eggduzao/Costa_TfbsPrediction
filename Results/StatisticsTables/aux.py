##################################################
## Auxiliary functions for table statistics
##################################################

# Import
import os
import sys
import aux
import glob
import Table
import numpy as np
from Bio import Motif

# TFBS / PWM Dictionaries
H1hescMPBS=["USF1_MA0093_2","BACH1_M00495","BRCA1_MA0133_1","CEBPB_MA0466_1","CTCF_MA0139_1",
"EGR1_MA0162_2","FOSL1_MA0477_1","GABP_MA0062_2","JUN_MA0488_1","JUND_MA0491_1", 
"MAFK_MA0496_1","MAX_MA0058_2","MYC_MA0147_2","NRF1_MA0506_1","P300_M00033", 
"POU5F1_MA0142_1","CTCF_MA0139_1","REST_MA0138_2","RFX5_MA0510_1","RXRA_MA0512_1", 
"ZNF143_MA0088_1","SP1_MA0079_3","SP2_MA0516_1","SP4_UP00002_1","SRF_MA0083_2", 
"TBP_MA0108_2","TCF12_MA0521_1","USF1_MA0093_2","USF2_MA0526_1","YY1_MA0095_2",
"ZNF143_MA0088_1"]
H1hescTFBS=["ATF3","BACH1","BRCA1","CEBPB","CTCF","EGR1","FOSL1","GABP","JUN","JUND",
"MAFK","MAX","MYC","NRF1","P300","POU5F1","RAD21","REST","RFX5","RXRA","SIX5","SP1","SP2","SP4","SRF",
"TBP","TCF12","USF1","USF2","YY1","ZNF143"]
HeLaS3MPBS=["BRCA1_MA0133_1","CEBPB_MA0466_1","CTCF_MA0139_1","E2F4_MA0470_1","E2F6_MA0471_1",
"ELK1_MA0028_1","FOS_MA0476_1","GABP_MA0062_2","JUN_MA0488_1","JUND_MA0491_1",
"MAFK_MA0496_1","MAX_MA0058_2","MYC_MA0147_2","NFYA_MA0060_2","NFYB_MA0502_1",
"NRF1_MA0506_1","P300_M00033","CTCF_MA0139_1","REST_MA0138_2","STAT1_MA0137_3",
"TBP_MA0108_2","USF2_MA0526_1","ZNF143_MA0088_1"]
HeLaS3TFBS=["BRCA1","CEBPB","CTCF","E2F4","E2F6","ELK1","FOS","GABP","JUN","JUND",
"MAFK","MAX","MYC","NFYA","NFYB","NRF1","P300","RAD21","REST","STAT1","TBP","USF2","ZNF143"]
HepG2MPBS=["ARID3A_MA0151_1","CREB1_MA0018_2","BHLHE40_MA0464_1","BRCA1_MA0133_1","CEBPB_MA0466_1",
"CTCF_MA0139_1","ELF1_MA0473_1","GABP_MA0062_2","JUN_MA0488_1","JUND_MA0491_1",
"MAFF_MA0495_1","MAFK_MA0496_1","MAX_MA0058_2","MYC_MA0147_2","NRF1_MA0506_1",
"P300_M00033","CTCF_MA0139_1","REST_MA0138_2","RXRA_MA0512_1","SP1_MA0079_3",
"SP2_MA0516_1","SRF_MA0083_2","TBP_MA0108_2","USF1_MA0093_2","USF2_MA0526_1",
"YY1_MA0095_2"]
HepG2TFBS=["ARID3A","ATF3","BHLHE40","BRCA1","CEBPB","CTCF","ELF1","GABP","JUN","JUND","MAFF","MAFK","MAX","MYC","NRF1",
"P300","RAD21","REST","RXRA","SP1","SP2","SRF","TBP","USF1","USF2","YY1"]
HuvecMPBS=["CTCF_MA0139_1","FOS_MA0476_1","GATA2_MA0036_2","JUN_MA0488_1","MAX_MA0058_2","MYC_MA0147_2"]
HuvecTFBS=["CTCF","FOS","GATA2","JUN","MAX","MYC"]
K562MPBS=["ARID3A_MA0151_1","ATF1_UP00020_1","CREB1_MA0018_2","BACH1_M00495","BHLHE40_MA0464_1",
"TAL1_MA0140_2","CEBPB_MA0466_1","CTCF_MA0139_1","CTCF_MA0139_1","E2F4_MA0470_1",
"E2F6_MA0471_1","FOS_MA0476_1","GATA2_MA0036_2","EGR1_MA0162_2","JUNB_MA0490_1",
"JUND_MA0491_1","ELF1_MA0473_1","ELK1_MA0028_1","ETS1_MA0098_2","FOS_MA0476_1",
"FOSL1_MA0477_1","GABP_MA0062_2","GATA1_MA0035_3","GATA2_MA0036_2","IRF1_MA0050_2",
"JUN_MA0488_1","JUND_MA0491_1","MAFF_MA0495_1","MAFK_MA0496_1","MAX_MA0058_2",
"MEF2A_MA0052_2","MYC_MA0147_2","NFE2_MA0501_1","NFYA_MA0060_2","NFYB_MA0502_1",
"NR2F2_UP00009_1","NRF1_MA0506_1","PU1_MA0080_3","CTCF_MA0139_1","REST_MA0138_2",
"RFX5_MA0510_1","ZNF143_MA0088_1","CTCF_MA0139_1","SP1_MA0079_3","SP2_MA0516_1",
"SRF_MA0083_2","STAT1_MA0137_3","STAT2_MA0517_1","STAT5A_MA0519_1","TAL1_MA0140_2",
"TBP_MA0108_2","THAP1_MA0597_1","TR4_MA0504_1","USF1_MA0093_2","USF2_MA0526_1",
"YY1_MA0095_2","ZBTB7A_UP00047_1","ZBTB33_MA0527_1","ZNF143_MA0088_1","ZNF263_MA0528_1"]
K562TFBS=["ARID3A","ATF1","ATF3","BACH1","BHLHE40","CCNT2","CEBPB","CTCF","CTCFL","E2F4","E2F6","EFOS","EGATA","EGR1","EJUNB", 
"EJUND","ELF1","ELK1","ETS1","FOS","FOSL1","GABP","GATA1","GATA2","IRF1","JUN","JUND","MAFF","MAFK","MAX",
"MEF2A","MYC","NFE2","NFYA","NFYB","NR2F2","NRF1","PU1","RAD21","REST","RFX5","SIX5","SMC3","SP1","SP2",
"SRF","STAT1","STAT2","STAT5A","TAL1","TBP","THAP1","TR4","USF1","USF2","YY1","ZBTB7A","ZBTB33","ZNF143","ZNF263"]
H1hescDict = dict(zip(H1hescTFBS,H1hescMPBS))
HeLaS3Dict = dict(zip(HeLaS3TFBS,HeLaS3MPBS))
HepG2Dict = dict(zip(HepG2TFBS,HepG2MPBS))
HuvecDict = dict(zip(HuvecTFBS,HuvecMPBS))
K562Dict = dict(zip(K562TFBS,K562MPBS))
pwmDict = dict([("H1hesc",H1hescDict),("HeLaS3",HeLaS3Dict),("HepG2",HepG2Dict),("Huvec",HuvecDict),("K562",K562Dict)])

# Get number of lines in file
def fileLen(fileName):
    i = -1
    with open(fileName) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Get number of lines in intersection
def fileLenIntersect(fileName1,fileName2,isReverse,tempFileName,tempExtName,totalExt=100):

    # Extending TFBS file
    tfbsFile = open(fileName1,"r")
    tempExt = open(tempExtName,"w")
    for line in tfbsFile:
        ll = line.strip().split("\t")
        tempExt.write("\t".join([ll[0],str(max(int(ll[1])-(totalExt/2),0)),str(int(ll[2])+(totalExt/2))])+"\n")
    tfbsFile.close()
    tempExt.close()

    # Performing intersection and calculating length
    if(isReverse): os.system("intersectBed -wa -a "+tempExtName+" -b "+fileName2+" -v | sort | uniq > "+tempFileName)
    else: os.system("intersectBed -wa -a "+tempExtName+" -b "+fileName2+" | sort | uniq > "+tempFileName)
    myLen = fileLen(tempFileName)
    os.system("rm "+tempFileName+" "+tempExtName)
    return myLen

# Creating table
# Output file must be .tex
def createTable(captionList, headerVec, dataMatrixList, outputFileName):
    latexFile = open(outputFileName,"w")
    latexFile.write("\\documentclass[landscape, 8pt]{report}\n\\usepackage{deluxetable}\n\\usepackage[landscape]{geometry}\n\\begin{document}\n")
    for i in range(0,len(dataMatrixList)):
        t = Table.Table(len(dataMatrixList[i]), justs="c"*len(dataMatrixList[i]), caption=captionList[i], label="tab:label"+str(i))
        t.add_header_row(headerVec)
        t.add_data(dataMatrixList[i], sigfigs=2)
        t.print_table(latexFile)
        latexFile.write("\\clearpage\n")
    latexFile.write("\\end{document}")
    latexFile.close()
    os.system("pdflatex "+outputFileName+" &> "+outputFileName+".log")
    lastName = outputFileName.split("/")[-1].split(".")[0]
    os.system("mv "+os.getcwd()+"/"+lastName+".pdf "+outputFileName[:-4]+".pdf")
    os.system("rm "+os.getcwd()+"/"+lastName+".aux "+os.getcwd()+"/"+lastName+".log "+outputFileName+".log")
    return 0

# Calculate bitscore
def calculateBitscore(pwmLoc,cell,factor,pseudocounts,precision,fpr,outputLocation):
    pwmFileName = pwmLoc+pwmDict[cell][factor]+".pfm"
    pwmFile = open(pwmFileName,"r")
    tempFileName = outputLocation+pwmFileName.split("/")[-1]+"temp"
    pwmFileT = open(tempFileName,"w")
    for line in pwmFile: pwmFileT.write(" ".join([str(float(e)+pseudocounts) for e in line.strip().split(" ")])+"\n")    
    pwmFile.close()
    pwmFileT.close()
    pwmFile = open(tempFileName,"r")
    pwm = Motif.read(pwmFile,"jaspar-pfm")
    pwmFile.close()
    os.system("rm "+tempFileName)
    sd = Motif.ScoreDistribution(pwm,precision=precision)
    return sd.threshold_fpr(fpr)


