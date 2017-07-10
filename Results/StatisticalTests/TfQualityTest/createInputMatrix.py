
# Import
import os
import sys
import glob
from Bio import Motif

# Classes to read Obo file
class OboElement:
  def __init__(self,name,oboId,subset):
    self.name = name
    self.oboId = oboId
    self.subset = subset
class OboList:
  def __init__(self,oboFileName,onlyGenus=False):
    self.fileName = oboFileName
    self.list = []
    oboFile = open(oboFileName,"r")
    flagStart = False; name = None; oboId = None; subset = None
    for line in oboFile:
      if(line[0] == "["):
        flagStart = True
        if(name):
          if((not onlyGenus) or (subset == "Genus")): self.list.append(OboElement(name,oboId,subset))
      if(flagStart):
        ll = line.strip().split(" ")
        lll = line[:3]
        if(lll == "egg"): name = ll[1]
        elif(lll == "id:"): oboId = ll[1]
        elif(lll == "sub"): subset = ll[1]
        elif(lll == "is_"): flagStart = False
    oboFile.close()
    if((not onlyGenus) or (subset == "Genus")): self.list.append(OboElement(name,oboId,subset))

  def get(self,factorName):
    ret = None
    for e in self.list:
      if(e.name == factorName):
        ret = e
        break
    return ret

# SCRIPT
if __name__ == '__main__':
  
  # Input
  oboFileName = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/Input/TFClass_format.obo"
  pwmLoc = "/home/egg/hpcwork/izkf/projects/egg/Data/PWM/Jaspar_Transfac/"
  rocLoc = "/home/egg/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/"
  factorFileName = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/TF.txt"
  tfbsStatisticsFileName = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/StatisticsTables/TFBS/TfbsStatisticsFinal.tex"
  outFileName = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/Input/table.txt"

  # Parameters
  orderBy = 5
  cellList = ["H1hesc","K562"]
  tfRemoveList = ["ARID3A","BRCA1","TBP","THAP1","P300","ZBTB7A"]
  H1hescMPBS=["USF1_MA0093_2","BACH1_M00495","BRCA1_MA0133_1","CEBPB_MA0466_1","CTCF_MA0139_1","EGR1_MA0162_2","FOSL1_MA0477_1",
              "GABP_MA0062_2","JUN_MA0488_1","JUND_MA0491_1","MAFK_MA0496_1","MAX_MA0058_2","MYC_MA0147_2","NRF1_MA0506_1",
              "P300_M00033","POU5F1_MA0142_1","CTCF_MA0139_1","REST_MA0138_2","RFX5_MA0510_1","RXRA_MA0512_1","ZNF143_MA0088_1",
              "SP1_MA0079_3","SP2_MA0516_1","SP4_UP00002_1","SRF_MA0083_2","TBP_MA0108_2","TCF12_MA0521_1","USF1_MA0093_2",
              "USF2_MA0526_1","YY1_MA0095_2","ZNF143_MA0088_1"]
  H1hescTFBS=["ATF3","BACH1","BRCA1","CEBPB","CTCF","EGR1","FOSL1","GABP","JUN","JUND","MAFK","MAX","MYC","NRF1","P300",
              "POU5F1","RAD21","REST","RFX5","RXRA","SIX5","SP1","SP2","SP4","SRF","TBP","TCF12","USF1","USF2","YY1","ZNF143"]
  K562MPBS=["ARID3A_MA0151_1","ATF1_UP00020_1","CREB1_MA0018_2","BACH1_M00495","BHLHE40_MA0464_1",
            "TAL1_MA0140_2","CEBPB_MA0466_1","CTCF_MA0139_1","CTCF_MA0139_1","E2F4_MA0470_1","E2F6_MA0471_1","FOS_MA0476_1",
            "GATA2_MA0036_2","EGR1_MA0162_2","JUNB_MA0490_1","JUND_MA0491_1","ELF1_MA0473_1","ELK1_MA0028_1","ETS1_MA0098_2",
            "FOS_MA0476_1","FOSL1_MA0477_1","GABP_MA0062_2","GATA1_MA0035_3","GATA2_MA0036_2","IRF1_MA0050_2","JUN_MA0488_1",
            "JUND_MA0491_1","MAFF_MA0495_1","MAFK_MA0496_1","MAX_MA0058_2","MEF2A_MA0052_2","MYC_MA0147_2","NFE2_MA0501_1",
            "NFYA_MA0060_2","NFYB_MA0502_1","NR2F2_UP00009_1","NRF1_MA0506_1","PU1_MA0080_3","CTCF_MA0139_1","REST_MA0138_2",
            "RFX5_MA0510_1","ZNF143_MA0088_1","CTCF_MA0139_1","SP1_MA0079_3","SP2_MA0516_1","SRF_MA0083_2","STAT1_MA0137_3",
            "STAT2_MA0517_1","STAT5A_MA0519_1","TAL1_MA0140_2","TBP_MA0108_2","THAP1_MA0597_1","TR4_MA0504_1","USF1_MA0093_2",
            "USF2_MA0526_1","YY1_MA0095_2","ZBTB7A_UP00047_1","ZBTB33_MA0527_1","ZNF143_MA0088_1","ZNF263_MA0528_1"]
  K562TFBS=["ARID3A","ATF1","ATF3","BACH1","BHLHE40","CCNT2","CEBPB","CTCF","CTCFL","E2F4","E2F6","EFOS","EGATA","EGR1","EJUNB", 
            "EJUND","ELF1","ELK1","ETS1","FOS","FOSL1","GABP","GATA1","GATA2","IRF1","JUN","JUND","MAFF","MAFK","MAX",
            "MEF2A","MYC","NFE2","NFYA","NFYB","NR2F2","NRF1","PU1","RAD21","REST","RFX5","SIX5","SMC3","SP1","SP2",
            "SRF","STAT1","STAT2","STAT5A","TAL1","TBP","THAP1","TR4","USF1","USF2","YY1","ZBTB7A","ZBTB33","ZNF143","ZNF263"]
  H1hescDict = dict(zip(H1hescTFBS,H1hescMPBS))
  K562Dict = dict(zip(K562TFBS,K562MPBS))
  pwmDict = dict([("H1hesc",H1hescDict),("K562",K562Dict)])

########################
# Creating Table
########################

  # Reading obo file
  obolist = OboList(oboFileName,onlyGenus=True)

  # Getting TF translation for tex file
  factorTranslation = dict()
  factorFile = open(factorFileName,"r")
  for line in factorFile:
    ll = line.strip().split("\t")
    factorTranslation[ll[0]] = ll[1]
  factorFile.close()

  # Iterating on cell types
  outTable = []
  for cell in cellList:

    # Iterating on factor names
    for aucFileName in glob.glob(rocLoc+cell+"/fdr_4/*_auc.txt"):

      # Getting factor name
      factorName = aucFileName.split("/")[-1].split("_")[0]
      if(factorName in tfRemoveList): continue

      # Getting TF classification
      obo = obolist.get(factorName)
      if(not obo): print factorName
      tfClass = obo.oboId
      tt = tfClass.split(".")

      # Getting AUCs
      aucFile = open(aucFileName,"r")
      aucDict = dict()
      for line in aucFile:
        ll = line.strip().split("\t")
        aucDict[ll[0]] = ll[1]
      aucFile.close()
      if(cell == "H1hesc"):
        aucHmm2 = aucDict["DH-HMM_13(H)"]
        aucHmm3 = aucDict["DH-HMM_139(H)"]
        aucHhm2 = aucDict["H-HMM_13(H)"]
        aucHhm3 = aucDict["H-HMM_139(H)"]
      elif(cell == "K562"):
        aucHmm2 = aucDict["DH-HMM_13(K)"]
        aucHmm3 = aucDict["DH-HMM_139(K)"]
        aucHhm2 = aucDict["H-HMM_13(K)"]
        aucHhm3 = aucDict["H-HMM_139(K)"]
      aucPwm = aucDict["PWM"]
      aucBoy = aucDict["Boyle"]
      aucNep = aucDict["Neph_5"]
      aucCue2 = aucDict["Cuellar_D13"]
      aucCue3 = aucDict["Cuellar_D139"]
      aucCen = aucDict["Pique_ces"]

      # Getting PWM IC
      pwmFileName = pwmLoc+pwmDict[cell][factorName]+".pfm"
      pwmFile = open(pwmFileName,"r")
      pwm = Motif.read(pwmFile,"jaspar-pfm")
      ic = str(pwm.ic())
      pwmFile.close()

      # Getting ChIP+PWM statistics
      tfbsStatisticsFile = open(tfbsStatisticsFileName,"r")
      flagStart1 = False; flagStart2 = False; counter = 0; chipStats = 0
      for line in tfbsStatisticsFile:
        ll = line.strip()
        if(ll == "\\label{tab:"+cell+".tfbsstats}"): flagStart1 = True
        if(flagStart1 and ("{"+factorTranslation[factorName]+"}" in ll)):
          flagStart2 = True
          counter = 0
          continue
        if(flagStart2): 
          if("\\multirow" not in ll):
            lll = line.split(" \\\\ ")[0].split(" & ")
            if(counter == 1): chipStats = lll[3]
            else: chipStats = lll[7]
            break
          counter += 1
      tfbsStatisticsFile.close()

      # Writing table
      outTable.append([cell,factorName,tt[0],tt[1],tt[2],tt[3],tt[4],ic,chipStats,
                       aucPwm,aucBoy,aucNep,aucCue2,aucCue3,aucCen,aucHmm2,aucHmm3,aucHhm2,aucHhm3])

  # Sorting output table
  outTable = sorted(outTable, key = lambda e: e[orderBy])

  # Writing table
  outFile = open(outFileName,"w")
  outFile.write("\t".join(["CELL","FACTOR","SUPERCLASS","CLASS","FAMILY","SUBFAMILY","GENUS","PWM_IC","CHIP_PWM",
                         "AUC_PWM","AUC_BOYLE","AUC_NEPH","AUC_CUE2","AUC_CUE3","AUC_CENT","AUC_DH2","AUC_DH3","AUC_H2","AUC_H3"])+"\n")
  for vec in outTable: outFile.write("\t".join(vec)+"\n")
  outFile.close()

  ########################
  # Separating Classes
  ########################

  # Function to create files separated by class
  def createClasses(keyVec,aucVec,outFileName,outReportName):

    # Separating by class
    classDict = dict()
    for vec in outTable:
      myKey = ".".join([vec[e] for e in keyVec])
      try:
        for e in aucVec: classDict[myKey].append(vec[e])
      except Exception:
        classDict[myKey] = [vec[e] for e in aucVec]

    # Insert NAs
    maxV = -1
    for k in classDict.keys():
      if(len(classDict[k]) > maxV): maxV = len(classDict[k])
    for k in classDict.keys():
      while(len(classDict[k]) < maxV): classDict[k].append("NA")

    # Writing table
    outFile = open(outFileName,"w")

    headerNames = classDict.keys()
    headerNames.sort()
    outFile.write("\t".join(headerNames)+"\n")
    for e in range(0,maxV):
      outFile.write(classDict[headerNames[0]][e])
      for k in headerNames[1:]:
        outFile.write("\t"+classDict[str(k)][e])
      outFile.write("\n")
    outFile.close()

    # Writing report
    outReport = open(outReportName,"w")
    for k in headerNames: outReport.write("\t".join([k,str(len([e for e in classDict[k] if (e != "NA")]))])+"\n")
    outReport.close()

  # Input
  outBoxLoc = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/Input/boxplot/"
  outReportLoc = "/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/Input/report/"

  # Parameters
  keyLabelList = ["superclass","class","family","subfamily","genus"]
  keyVecList = [[2],[2,3],[2,3,4],[2,3,4,5],[2,3,4,5,6]]
  aucLabelList = ["pwm","boyle","neph","cuellar","centipede","dh-hmm","h-hmm"]
  aucVecList = [[9],[10],[11],[13],[14],[16],[18]]

  # Execute function
  for i in range(0,len(keyVecList)):
    for j in range(0,len(aucVecList)):
      keyVec = keyVecList[i]
      aucVec = aucVecList[j]
      outName = keyLabelList[i]+"_"+aucLabelList[j]+".txt"
      outFileName = outBoxLoc+outName
      outReportName = outReportLoc+outName
      createClasses(keyVec,aucVec,outFileName,outReportName)


