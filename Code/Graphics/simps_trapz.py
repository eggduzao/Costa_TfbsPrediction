
import os
import sys

###############################################################################
# FUNCTION GET RATES
###############################################################################

def get_rates(bedFileName, tfb, rateDict):

  # Default BCTYPE
  myName = bedFileName.split("/")[-1].split(".")[0]
  sbRate, myStep, bcType = rateDict[myName]

  # Bias Correction
  if(bcType == "B"): myRate = sbRate
  elif(bcType == "BC"): myRate = sbRate+(tfb*0.032)-0.005
  elif(bcType == "BCN"): myRate = sbRate+(tfb*0.022)-0.005
  else: myRate = sbRate

  return myRate, myStep

###############################################################################
# FUNCTION SIMP
###############################################################################

def simp(mpbsName, cellType, myLabel, bedFileName, sortedFileName, rateDict):

  #################################################
  # TFB
  #################################################

  # Default TFB
  tfb = 0.9
  if(mpbsName == "ARID3A"): tfb = abs(float(0.80716421887))
  elif(mpbsName == "ATF1"): tfb = abs(float(0.687256109933))
  elif(mpbsName == "ATF3"): tfb = abs(float(0.612195738469))
  elif(mpbsName == "BACH1"): tfb = abs(float(0.461480034821))
  elif(mpbsName == "BHLHE40"): tfb = abs(float(0.563904607907))
  elif(mpbsName == "CCNT2"): tfb = abs(float(0.620201438553))
  elif(mpbsName == "CEBPB"): tfb = abs(float(0.837443979326))
  elif(mpbsName == "CTCF"): tfb = abs(float(0.226450856972))
  elif(mpbsName == "CTCFL"): tfb = abs(float(0.196114634764))
  elif(mpbsName == "E2F4"): tfb = abs(float(0.518568752189))
  elif(mpbsName == "E2F6"): tfb = abs(float(0.625726742361))
  elif(mpbsName == "EFOS"): tfb = abs(float(0.755693271069))
  elif(mpbsName == "EGATA"): tfb = abs(float(0.796900563532))
  elif(mpbsName == "EGR1"): tfb = abs(float(0.763513484225))
  elif(mpbsName == "EJUNB"): tfb = abs(float(0.69021680169))
  elif(mpbsName == "EJUND"): tfb = abs(float(0.737108537119))
  elif(mpbsName == "ELF1"): tfb = abs(float(0.308332265108))
  elif(mpbsName == "ELK1"): tfb = abs(float(0.455771814968))
  elif(mpbsName == "ETS1"): tfb = abs(float(0.585564334152))
  elif(mpbsName == "FOS"): tfb = abs(float(0.738568124619))
  elif(mpbsName == "FOSL1"): tfb = abs(float(0.752207666107))
  elif(mpbsName == "GABP"): tfb = abs(float(0.463500531325))
  elif(mpbsName == "GATA1"): tfb = abs(float(0.838360666041))
  elif(mpbsName == "GATA2"): tfb = abs(float(0.803147857326))
  elif(mpbsName == "IRF1"): tfb = abs(float(0.876927353785))
  elif(mpbsName == "JUN"): tfb = abs(float(0.70375016811))
  elif(mpbsName == "JUND"): tfb = abs(float(0.744622666669))
  elif(mpbsName == "MAFF"): tfb = abs(float(0.627273230589))
  elif(mpbsName == "MAFK"): tfb = abs(float(0.607781675024))
  elif(mpbsName == "MAX"): tfb = abs(float(0.45307600016))
  elif(mpbsName == "MEF2A"): tfb = abs(float(0.817226869474))
  elif(mpbsName == "MYC"): tfb = abs(float(0.266216093732))
  elif(mpbsName == "NFE2"): tfb = abs(float(0.563238926177))
  elif(mpbsName == "NFYA"): tfb = abs(float(0.4442794763))
  elif(mpbsName == "NFYB"): tfb = abs(float(0.39409043756))
  elif(mpbsName == "NR2F2"): tfb = abs(float(0.592578609471))
  elif(mpbsName == "NRF1"): tfb = abs(float(0.358511386411))
  elif(mpbsName == "PU1"): tfb = abs(float(0.560204927134))
  elif(mpbsName == "RAD21"): tfb = abs(float(0.146943213521))
  elif(mpbsName == "REST"): tfb = abs(float(0.610249613624))
  elif(mpbsName == "RFX5"): tfb = abs(float(0.5817831067))
  elif(mpbsName == "SIX5"): tfb = abs(float(0.631474006345))
  elif(mpbsName == "SMC3"): tfb = abs(float(0.162677324475))
  elif(mpbsName == "SP1"): tfb = abs(float(0.431213014296))
  elif(mpbsName == "SP2"): tfb = abs(float(0.632100986497))
  elif(mpbsName == "SRF"): tfb = abs(float(0.736995307761))
  elif(mpbsName == "STAT1"): tfb = abs(float(0.530256300528))
  elif(mpbsName == "STAT2"): tfb = abs(float(0.85211703434))
  elif(mpbsName == "STAT5A"): tfb = abs(float(0.639081002969))
  elif(mpbsName == "TAL1"): tfb = abs(float(0.655570417059))
  elif(mpbsName == "TBP"): tfb = abs(float(0.658562471446))
  elif(mpbsName == "THAP1"): tfb = abs(float(0.676301280561))
  elif(mpbsName == "TR4"): tfb = abs(float(0.679182194593))
  elif(mpbsName == "USF1"): tfb = abs(float(0.587444376029))
  elif(mpbsName == "USF2"): tfb = abs(float(0.526198069526))
  elif(mpbsName == "YY1"): tfb = abs(float(0.720753283229))
  elif(mpbsName == "ZBTB33"): tfb = abs(float(0.469969486229))
  elif(mpbsName == "ZBTB7A"): tfb = abs(float(0.905862807171))
  elif(mpbsName == "ZNF143"): tfb = abs(float(0.661591891414))
  elif(mpbsName == "ZNF263"): tfb = abs(float(0.828219568556))
  else: tfb = 0.9

  #################################################
  # Evaluating rateN
  #################################################

  # Counting N
  sortFile = open(sortedFileName,"r")
  counterN = 0
  for line in sortFile:
    ll = line.strip().split("\t")
    lll = ll[3].split(":")
    if(lll[0] != mpbsName): continue
    elif(lll[1] != "Y"): counterN += 1
  sortFile.close()

  # Evaluating rateN
  myRate, myStep = get_rates(bedFileName, tfb, rateDict)
  rateN = int(myRate*counterN)

  #################################################
  # Evaluating based on trapz simps
  #################################################

  # Fixing
  sortFile = open(sortedFileName,"r")
  oFileName = sortedFileName+"_TEMP.bed"
  oFile = open(oFileName,"w")
  counterN = 0; counter2 = 0
  for line in sortFile:
    ll = line.strip().split("\t")
    lll = ll[3].split(":")
    if(lll[0] != mpbsName): continue
    elif(lll[1] != "Y"): 
      if((counterN < rateN) and (counter2%myStep != 0)):
        ll[4] = "0"
        counterN += 1
      counter2 += 1
    oFile.write("\t".join(ll)+"\n")
  sortFile.close()
  oFile.close()
  oFileName2 = oFileName+"_TEMP.bed"
  os.system("sort -r -k5 -g "+oFileName+" > "+oFileName2)
  os.system("rm "+sortedFileName+" "+oFileName)
  sortedFileName = oFileName2

  return sortedFileName


