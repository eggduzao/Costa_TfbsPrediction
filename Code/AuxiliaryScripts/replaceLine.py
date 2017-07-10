#################################################################################################
# Replaces strings into whole file.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. oldString = The old string to be replaced.")
params.append("    2. newString = The new string to be written.")
params.append("    3. [inputList] = List of files.")
params.append("###")
params.append("Output: ")
params.append("    1. [outputList] = List of files with oldString replaced by newString.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
oldString = sys.argv[1]
newString = sys.argv[2]
inputList = sys.argv[3:]

# Replacing strings
for inputFileName in inputList:
    inputFile = open(inputFileName,"r")
    outputFile = open(inputFileName+"temp","w")
    for line in inputFile:
        if(oldString in line): line = line.replace(oldString,newString)
        outputFile.write(line)
    inputFile.close()
    outputFile.close()
    os.system("rm "+inputFileName)
    os.system("mv "+inputFileName+"temp "+inputFileName)


