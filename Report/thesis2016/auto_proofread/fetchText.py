
import os
import sys

# Parameters
initPage = "3"
finalPage = "4"
pdfFileName = "../thesis.pdf"
textFileName = "thesis.txt"
outTextFileName = "thesis_corr.txt"
outFileName = "correction.txt"

# First text conversion
os.system("cp ../thesis.pdf ./thesis.pdf")
os.system("pdftotext -nopgbrk -eol unix -f "+initPage+" -l "+finalPage+" thesis.pdf")
os.system("rm thesis.pdf")

# Clearing text
inFile = open(textFileName,"r")
outFile = open(outTextFileName,"w")
for i in range(0,5): inFile.readline()
prev = ""
for line in inFile:
  if(prev == "\n" and "." in line): outFile.write("\n")
  else: outFile.write(line.strip()+" ")
  prev = line
inFile.close()
outFile.close()
os.system("rm "+textFileName)

# Making conversion
os.system("java -jar ./LanguageTool-3.3/languagetool-commandline.jar -l en-US "+outTextFileName+" > "+outFileName)
os.system("rm "+outTextFileName)


