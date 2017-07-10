
# Import
import os
import sys

# Input
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/"
javaFileName = "StatisticTestStats"
wekaFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/weka.jar"
outputFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/resultsStats.txt"
texFileName = "resultsStats"
headerFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/Input/headerStats.txt"

# Parameters
methodVec = ["ac","ba","sn","sp","pp","np"]
fencyVec = ["Ac","Ba","Sn","Sp","Pp","Np"]
fencyMethod = dict([(methodVec[i],fencyVec[i]) for i in range(0,len(methodVec))])

# Running test
os.system("javac -classpath "+wekaFileName+" "+inLoc+javaFileName+".java")
os.system("java -classpath "+inLoc+":"+wekaFileName+" "+javaFileName+" > "+outputFileName)

# Fetching method names
methodNameVec = []
headerFile = open(headerFileName,"r")
for line in headerFile:
    methodNameVec.append(line.strip())
headerFile.close()

# Creating tex
texFile = open(inLoc+texFileName+".tex","w")
texFile.write("\\documentclass[landscape, 8pt]{report}\n")
texFile.write("\\usepackage[landscape]{geometry}\n")
texFile.write("\\usepackage{rotating}\n")
texFile.write("\\usepackage{multirow}\n")
texFile.write("\\begin{document}\n\n")

# Iterating on test results
inFile = open(outputFileName,"r")
currMethod = None
rankingList = []
tableList = []
flagRank = False
flagTable = False
for line in inFile:
    
    # New Method
    if("Estatisticas -- " in line):
        if(currMethod):
            rankingList.append(rankingVec)
            tableList.append(table)
        currMethod = line.strip().split("/")[-1].split(".")[0]
        rankingVec = []
        table = []

    # Fetching ranking
    if(flagRank):
        if("Friedman" in line):
            flagRank = False
            continue
        rankingVec.append(line.strip().split("\t"))
    elif(line.strip() == "Ranking:"): flagRank = True

    # Fetching table
    if(flagTable):
        if("Win/Tie/Loss" in line):
            flagTable = False
            continue
        elif(line[0] == ","): continue
        ll = line.strip().split(",")
        table.append(ll[1:-1])
    elif("Tabela de Comparacao" in line): flagTable = True

# Last results
rankingList.append(rankingVec)
tableList.append(table)

# Writing Ranking
texFile.write("\\begin{table}[h!]\n")
texFile.write("\\label{tab:ranking}\n")
texFile.write("\\vspace{0.0cm}\n")
texFile.write("\\begin{center}\n")
texFile.write("\\caption{Friedman ranking. For each metric, the methods are displayed in decreasing order with their respective Friedman ranking.}\n")
texFile.write("\\renewcommand{\\arraystretch}{1.2}\n")
texFile.write("  \\begin{tabular}{ |lr|lr|lr|lr|lr|lr| }\n")
texFile.write("    \\hline\n")
texFile.write("    "+" & ".join(["\multicolumn{2}{|c|}{\\textbf{"+e+"}}" for e in fencyVec])+" \\\\\n")
texFile.write("    \\hline\n")
for i in range(0,len(rankingList[0])):
    for j in range(0,len(rankingList)):
        texFile.write("    ")
        final = " & "
        if(j == len(rankingList)-1): final = " \\\\\n"
        texFile.write(methodNameVec[int(rankingList[j][i][0])]+" & "+str(round(float(rankingList[j][i][1]),4))+final)
texFile.write("    \\hline\n")
texFile.write("  \\end{tabular}\n")
texFile.write("\\end{center}\n")
texFile.write("\\vspace{0.0cm}\n")
texFile.write("\\end{table}\n\n")

# Writing Tables
for i in range(0,len(tableList)):
    texFile.write("\\begin{table}[h!]\n")
    texFile.write("\\label{tab:friedman.nemenyi."+methodVec[i]+"}\n")
    texFile.write("\\vspace{0.0cm}\n")
    texFile.write("\\begin{center}\n")
    texFile.write("\\caption{Friedman-Nemenyi hypothesis test results for the \\textbf{"+fencyVec[i]+"} metric. The asterisc and the cross, respectively, mean that the method in the column outperformed the method in the row with significance levels of 0.05 and 0.1}\n")
    texFile.write("\\vspace{0.5cm}\n")
    texFile.write("\\renewcommand{\\arraystretch}{1.2}\n")
    texFile.write("  \\begin{tabular}{ rccccccccc }\n")
    #texFile.write("    \\hline\n")
    texFile.write("    & "+" & ".join(["\\rotatebox{90}{"+methodNameVec[int(rankingList[i][e][0])]+"}" for e in range(0,len(tableList[i]))])+" \\\\\n")
    texFile.write("    \\hline\n")

    for m in range(0,len(tableList[i])):
        sortIndexM = int(rankingList[i][m][0])
        toWriteVec = [methodNameVec[sortIndexM]]
        for n in range(0,len(tableList[i][sortIndexM])):
            sortIndexN = int(rankingList[i][n][0])
            s = "   "
            if(tableList[i][sortIndexM][sortIndexN] == "*"): s  = "$*$"
            elif(tableList[i][sortIndexM][sortIndexN] == "+"):  s = "$+$"
            toWriteVec.append(s)
        texFile.write("    "+" & ".join(toWriteVec)+" \\\\\n")

    texFile.write("    \\hline\n")
    texFile.write("  \\end{tabular}\n")
    texFile.write("\\end{center}\n")
    texFile.write("\\vspace{0.0cm}\n")
    texFile.write("\\end{table}\n\n")

# Writing all tables in one
texFile.write("\\begin{table}[h!]\n")
texFile.write("\\label{tab:friedman.nemenyi}\n")
texFile.write("\\vspace{0.0cm}\n")
texFile.write("\\begin{center}\n")
texFile.write("\\caption{Friedman-Nemenyi hypothesis test results for all metrics. The asterisc and the cross, respectively, mean that the method in the column outperformed the method in the row with significance levels of 0.05 and 0.1}\n")
texFile.write("\\renewcommand{\\arraystretch}{1.2}\n")
texFile.write("  \\begin{tabular}{ llccccccccc }\n")
texFile.write("  \\hline\n")
texFile.write("  & & \\begin{sideways}DH-HMM(3)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}DH-HMM(2)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}Cent.(estimated)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}Neph\\end{sideways} &\n")
texFile.write("  \\begin{sideways}Cuellar(2)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}Cuellar(3)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}H-HMM(3)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}H-HMM(2)\\end{sideways} &\n")
texFile.write("  \\begin{sideways}Boyle\\end{sideways} \\\\\n")
texFile.write("  \\hline\n")

for i in range(0,len(tableList)):
    if(fencyVec[i] not in ["Sn","Sp","AUC"]): continue
    texFile.write("    \\multirow{9}{*}{\\begin{sideways}"+fencyVec[i]+"\\end{sideways}}\n")
    for m in range(0,len(tableList[i])):
        toWriteVec = [methodNameVec[m]]
        for n in range(0,len(tableList[i][m])):
            s = "   "
            if(tableList[i][m][n] == "*"): s  = "$*$"
            elif(tableList[i][m][n] == "+"):  s = "$+$"
            toWriteVec.append(s)
        texFile.write("    & "+" & ".join(toWriteVec)+" \\\\\n")
    texFile.write("  \\hline\n")

texFile.write("  \\end{tabular}\n")
texFile.write("\\end{center}\n")
texFile.write("\\vspace{0.0cm}\n")
texFile.write("\\end{table}\n\n")

# Termination
texFile.write("\\end{document}\n")
inFile.close()
texFile.close()
os.system("pdflatex "+inLoc+texFileName+".tex")
os.system("rm "+inLoc+texFileName+".aux "+inLoc+texFileName+".log")


