
# Import
import os
import sys

# Input
inputFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TableStatistics/TFBS/TfbsStatistics.tex"
outputFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TableStatistics/TFBS/TfbsStatisticsFinal.tex"
factorFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/TF.txt"
inputFile = open(inputFileName,"r")
outputFile = open(outputFileName,"w")
cellList = ["H1hesc","HeLaS3","HepG2","K562"]

# Reading factor translation
factorTranslation = dict()
factorFile = open(factorFileName,"r")
for line in factorFile:
    ll = line.strip().split("\t")
    factorTranslation[ll[0]] = ll[1]
factorFile.close()

# Fetching data
dataDict = dict() # Cell -> ev -> [factor,bitscore,mpbs,chip,chip+mpbs]
currCell = ""
currEv = ""
flagStart = False
for line in inputFile:
    
    if("\\tablecaption{" in line):
        ll = line.strip().split("\\tablecaption{")[-1].split("\\label{tab:")[0].split(" ")
        currCell = ll[0]
        currEv = ll[1]+"_"+ll[2]
        try: dataDict[currCell][currEv] = []
        except Exception: dataDict[currCell] = dict([(currEv,[])])
    elif("\\startdata" in line):
        flagStart = True
        continue
    elif("\\enddata" in line): flagStart = False

    if(flagStart):
        ll = line[:-3].split(" & ")

        # Filtering factors
        if(currCell == "H1hesc"):
            if(ll[0] in ["BRCA1","FOSL1","P300","TBP"]): continue
        elif(currCell == "HeLaS3"):
            if(ll[0] in ["BRCA1","P300","TBP"]): continue
        elif(currCell == "HepG2"):
            if(ll[0] in ["ARID3A","ATF3","BRCA1","P300","TBP"]): continue
        elif(currCell == "K562"):
            if(ll[0] in ["ARID3A","TBP","THAP1","ZBTB7A"]): continue

        if(ll[1] == "--"): dataDict[currCell][currEv].append([ll[0],ll[1],ll[2],ll[3],ll[-1]])
        else: dataDict[currCell][currEv].append([ll[0],str(round(float(ll[1]),4)),ll[2],ll[3],ll[-1].split(" ")[1][1:-1]])

# Writing tex start
outputFile.write("\\documentclass[landscape, 8pt]{report}\n")
outputFile.write("\\usepackage{rotating}\n")
outputFile.write("\\usepackage{multirow}\n")
outputFile.write("\\usepackage{array}\n")
outputFile.write("\\begin{document}\n\n")

# Writing data
for cell in cellList:

    outputFile.write("\\begin{table}[t]\n")
    outputFile.write("\\begin{center}\n")
    outputFile.write("\\caption{Statistics for "+cell+" gold standard dataset.}\n")
    outputFile.write("\\label{tab:"+cell+".tfbsstats}\n")
    outputFile.write("    \\renewcommand{\\arraystretch}{1.2}\n")
    outputFile.write("    \\begin{tabular}{")

    outputFile.write(" |>{\\centering\\arraybackslash} m{1.8cm} >{\\centering\\arraybackslash} m{1.2cm} >{\\centering\\arraybackslash} m{1.4cm} >{\\centering\\arraybackslash} m{1.4cm} >{\\centering\\arraybackslash} m{1.6cm} | >{\\centering\\arraybackslash} m{1.8cm} >{\\centering\\arraybackslash} m{1.2cm} >{\\centering\\arraybackslash} m{1.4cm} >{\\centering\\arraybackslash} m{1.4cm} >{\\centering\\arraybackslash} m{1.6cm} | }\n")

    outputFile.write("        \\hline\n")
    outputFile.write("        \\textbf{Factor} & \\textbf{ChIP-seq Peaks} & \\textbf{Bit-score} & \\textbf{MPBSs} & \\textbf{ChIP+ MPBS(\\%)} & \\textbf{Factor} & \\textbf{ChIP-seq Peaks} & \\textbf{Bit-score} & \\textbf{MPBSs} & \\textbf{ChIP+ MPBS(\\%)} \\\\\n")
    outputFile.write("        \\hline\n")

    for i in range(0,len(dataDict[cell]["fdr_4"]),2):
        if(i+1 < len(dataDict[cell]["fdr_4"])):

            outputFile.write("        \\multirow{2}{*}{"+factorTranslation[dataDict[cell]["fdr_4"][i][0]]+"} & ")
            outputFile.write("\\multirow{2}{*}{"+dataDict[cell]["fdr_4"][i][3]+"} & ")
            outputFile.write("13.2877 & "+dataDict[cell]["bitscore_log"][i][2]+" & "+dataDict[cell]["bitscore_log"][i][4]+" & \n")

            outputFile.write("        \\multirow{2}{*}{"+factorTranslation[dataDict[cell]["fdr_4"][i+1][0]]+"} & ")
            outputFile.write("\\multirow{2}{*}{"+dataDict[cell]["fdr_4"][i+1][3]+"} & ")
            outputFile.write("13.2877 & "+dataDict[cell]["bitscore_log"][i+1][2]+" & "+dataDict[cell]["bitscore_log"][i+1][4]+" \\\\ \n")

            outputFile.write("        & & "+dataDict[cell]["fdr_4"][i][1]+" & "+dataDict[cell]["fdr_4"][i][2]+" & "+dataDict[cell]["fdr_4"][i][4])
            outputFile.write(" & & & "+dataDict[cell]["fdr_4"][i+1][1]+" & "+dataDict[cell]["fdr_4"][i+1][2]+" & "+dataDict[cell]["fdr_4"][i+1][4]+" \\\\ ")
            outputFile.write("\\hline\n")

            #outputFile.write("        \\multirow{2}{*}{"+factorTranslation[dataDict[cell]["fdr_4"][i][0]]+"} & ")
            #outputFile.write("\\multirow{2}{*}{"+dataDict[cell]["fdr_4"][i][3]+"} & ")
            #outputFile.write("13.2877 & "+dataDict[cell]["bitscore_log"][i][2]+" & "+dataDict[cell]["bitscore_log"][i][4]+" \\\\ ")
            #outputFile.write("& & ")
            #outputFile.write(dataDict[cell]["fdr_4"][i][1]+" & "+dataDict[cell]["fdr_4"][i][2]+" & "+dataDict[cell]["fdr_4"][i][4]+" \\\\ ")
            #outputFile.write("\\hline\n")
        else:
            outputFile.write("        \\multirow{2}{*}{"+factorTranslation[dataDict[cell]["fdr_4"][i][0]]+"} & ")
            outputFile.write("\\multirow{2}{*}{"+dataDict[cell]["fdr_4"][i][3]+"} & ")
            outputFile.write("13.2877 & "+dataDict[cell]["bitscore_log"][i][2]+" & "+dataDict[cell]["bitscore_log"][i][4]+" & \n")

            outputFile.write("        \\multirow{2}{*}{} & \\multirow{2}{*}{} & & & \\\\ \n")

            outputFile.write("        & & "+dataDict[cell]["fdr_4"][i][1]+" & "+dataDict[cell]["fdr_4"][i][2]+" & "+dataDict[cell]["fdr_4"][i][4])
            outputFile.write(" & & & & & \\\\ ")
            outputFile.write("\\hline\n")

    outputFile.write("    \\end{tabular}\n")
    outputFile.write("\\end{center}\n")
    outputFile.write("\\end{table}\n")
    outputFile.write("\\clearpage\n")

# Writing tex end
outputFile.write("\n\\end{document}\n")

# Termination
inputFile.close()
outputFile.close()


