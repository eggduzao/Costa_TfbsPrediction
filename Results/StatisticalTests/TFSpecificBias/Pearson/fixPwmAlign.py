
# Import
import os
import sys
from glob import glob

# Input
inList = glob("./*/PWM/*.pwm")
#aDict = dict([("NRF1",-1), ("SP4",-1), ("RAD21",-2), ("MAX",-1), ("CTCF",-2), ("MYC",-1), ("BRCA1",0), 
#              ("REST",-2), ("USF2",-3), ("YY1",-4), ("USF1",-3), ("ATF3",-3), ("RXRA",-1), 
#              ("GABP",0), ("RFX5",-1), ("P300",-1), ("TCF12",0), ("SIX5",1), ("JUN",0), 
#              ("ZNF143",1), ("SP2",-2), ("FOSL1",-2), ("CEBPB",0), ("SRF",-3), ("POU5F1",0), 
#              ("JUND",-2), ("BACH1",-1), ("MAFK",-2), ("TBP",2), ("SP1",-2), ("EGR1",-2), 
#              ("ELK1",-3), ("NFYB",-3), ("NFYA",-4), ("SMC3",0), ("CTCFL",0), ("ELF1",-3), 
#              ("ZBTB33",0), ("STAT1",-1), ("NFE2",0), ("STAT5A",1), ("E2F4",-1), ("PU1",-3), 
#              ("CCNT2",3), ("ETS1",-1), ("BHLHE40",-1), ("TAL1",3), ("NR2F2",-1), ("E2F6",-2), 
#              ("ATF1",-1), ("MAFF",-2), ("TR4",2), ("THAP1",-1), ("EGATA",-3), ("GATA2",-3), 
#              ("ZNF263",0), ("STAT2",1), ("EJUNB",-2), ("FOS",0), ("GATA1",-3), ("EJUND",0), 
#              ("EFOS",0), ("MEF2A",-2), ("ARID3A",-2), ("IRF1",1), ("ZBTB7A",0)])

aDict = dict([("NRF1",1), ("SP4",2), ("RAD21",0), ("MAX",0), ("CTCF",0), ("MYC",0), ("BRCA1",0), 
              ("REST",0), ("USF2",-1), ("YY1",-2), ("USF1",-2), ("ATF3",-1), ("RXRA",3), 
              ("GABP",-1), ("RFX5",0), ("P300",3), ("TCF12",2), ("SIX5",0), ("JUN",0), 
              ("ZNF143",-1), ("SP2",0), ("FOSL1",0), ("CEBPB",3), ("SRF",-1), ("POU5F1",2), 
              ("JUND",0), ("BACH1",-1), ("MAFK",-1), ("TBP",0), ("SP1",0), ("EGR1",0), 
              ("ELK1",-1), ("NFYB",-2), ("NFYA",0), ("SMC3",1), ("CTCFL",1), ("ELF1",-2), 
              ("ZBTB33",1), ("STAT1",0), ("NFE2",5), ("STAT5A",0), ("E2F4",1), ("PU1",-2), 
              ("CCNT2",5), ("ETS1",0), ("BHLHE40",0), ("TAL1",5), ("NR2F2",3), ("E2F6",0), 
              ("ATF1",0), ("MAFF",0), ("TR4",0), ("THAP1",0), ("EGATA",-2), ("GATA2",-2), 
              ("ZNF263",0), ("STAT2",2), ("EJUNB",-1), ("FOS",0), ("GATA1",0), ("EJUND",0), 
              ("EFOS",0), ("MEF2A",-2), ("ARID3A",0), ("IRF1",-3), ("ZBTB7A",2), 
              ("ER_0min",-3), ("ER_2min",-3), ("ER_5min",-3), ("ER_10min",-3), ("ER_40min",-3), ("ER_160min",-3),
              ("P53",0), 
              ("UW_Motif_0458",0), ("UW_Motif_0500",-1), 
              ("AR_1nmR",-1), ("AR_10nmR",-1), ("AR_ethl",-1), ("AR_R1881",-1), 
              ("GR_withDEX",0), ("GR_woDEX",0)])

# Execution
for inFileName in inList:
  if("_align" in inFileName.split("/")[-1]): continue
  factorName = inFileName.split("/")[-1].split(".")[0]
  inFile = open(inFileName,"r")
  pwm = []
  for line in inFile: pwm.append(line.strip().split(" "))
  inFile.close()
  if(aDict[factorName] > 0):
    pwmAdd = [[pwm[i][0]]*abs(aDict[factorName]) for i in range(0,4)]
    newPwm = []
    for i in range(0,len(pwm)): newPwm.append(pwmAdd[i]+pwm[i][:-aDict[factorName]])
  elif(aDict[factorName] < 0):
    pwmAdd = [[pwm[i][-1]]*abs(aDict[factorName]) for i in range(0,4)]
    newPwm = []
    for i in range(0,len(pwm)): newPwm.append(pwm[i][abs(aDict[factorName]):]+pwmAdd[i])
  else: newPwm = pwm

  outFileName = inFileName[:-4]+"_align.pwm"
  outFile = open(outFileName,"w")
  for vec in newPwm: outFile.write(" ".join(vec)+"\n")
  outFile.close()


