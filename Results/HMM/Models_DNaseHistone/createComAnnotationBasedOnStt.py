
# Import
import os
import sys
import glob

# Input
inputList = [e.split("/")[-1].split(".")[0] for e in glob.glob("./stt/*.stt")]
sttLocation = "./stt/"
bedLocation = "./bed/"

# Conversion dictionaries
sttDict = dict([("0","0"),("1","1"),("2","1"),("3","1"),("4","2"),("5","2"),("6","2"),("7","3")])
bedNameDict = dict([("BACK","BACK"), ("UPH","HH"), ("TOPH","HH"), ("DOWNH","HH"), ("UPD","HD"), ("TOPD","HD"), ("DOWND","HD"), ("FP","FP")])
bedColorDict = dict([("50,50,50","50,50,50"), ("110,250,110","90,180,240"), ("90,180,240","90,180,240"), ("255,80,90","90,180,240"), ("10,80,0","10,80,0"), ("20,40,150","10,80,0"), ("150,20,40","10,80,0"), ("198,150,0","198,150,0")])

# Creating stt files
for inputName in inputList:
    inputFileName = sttLocation+inputName+".stt"
    outputFileName = sttLocation+inputName+"_com.stt"
    inputFile = open(inputFileName,"r")
    outputFile = open(outputFileName,"w")
    for line in inputFile:
        ll = line.strip().split(" ")
        newAnnotation = "".join([sttDict[k] for k in ll[3]])
        outputFile.write(" ".join(ll[:3]+[newAnnotation])+"\n")
    inputFile.close()
    outputFile.close()

# Creating bed files
for inputName in inputList:
    inputFileName = bedLocation+inputName+".bed"
    outputFileName = bedLocation+inputName+"_com.bed"
    inputFile = open(inputFileName,"r")
    outputFile = open(outputFileName,"w")
    for line in inputFile:
        ll = line.strip().split(" ")
        outputFile.write(" ".join(ll[:3]+[bedNameDict[ll[3]]]+ll[4:8]+[bedColorDict[ll[8]]])+"\n")
    inputFile.close()
    outputFile.close()


