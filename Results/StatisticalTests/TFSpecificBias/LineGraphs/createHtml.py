
# Import
import os
import sys
from glob import glob

# Input
il="/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/LineGraphs/"
pl="/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
inList = ["DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562"]

# Execution
for inName in inList:

  # Fetching Pearson
  pFile = open(pl+inName+"/pearson.txt","r")
  pFile.readline()
  pTable = []
  for line in pFile:
    ll = line.strip().split("\t")
    pTable.append([ll[0],float(ll[1])]) # 50bp pearson
    #pTable.append([ll[0],float(ll[3])]) # motif len pearson
  pFile.close()
  pTable = sorted(pTable, key=lambda x:x[1], reverse=True)

  # Writing html
  outputFileName = "./"+inName+".html"
  outputFile = open(outputFileName,"w")
  outputFile.write("<html>\n<body>\n<table border=\"1\" style=\"width:30%\" align=\"center\">\n  <tr>\n    <th>Factor</th>\n    <th>Rank</th>\n    <th>Line Graph</th>\n    <th>Pearson Correlation</th>\n  </tr>\n\n")
  count = 1
  total = len(pTable)
  for e in pTable:
    factorName = e[0]
    factorPearson = str(round(e[1],2))
    rank = str(count)+"/"+str(total)
    count += 1
    factorImageLoc = il+inName+"/"+factorName
    os.system("convert "+factorImageLoc+"_pwm.eps "+factorImageLoc+"_pwm.png")
    outputFile.write("  <tr>\n    <td>"+factorName+"</td>\n    <td>"+rank+"</td>\n    <td><img src=\""+factorImageLoc+"_pwm.png\" height=\"500\" width=\"500\"></td>\n    <td>"+factorPearson+"</td>\n  </tr>")
  outputFile.write("\n</table>\n</body>\n</html>")
  outputFile.close()


