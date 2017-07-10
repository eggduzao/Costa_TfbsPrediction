
# Import
import os
import sys

# Input
jasparLocation = "/home/egg/Desktop/PWM/Jaspar/"
outputLocation = "/home/egg/Desktop/PWM/MEME/"

# Converting jaspar to meme
memeFileName = outputLocation+"all.meme"
os.system("jaspar2meme -pfm "+jasparLocation+" > "+memeFileName)

# Splitting meme file
memeFile = open(memeFileName,"r")
outputFile = None
for line in memeFile:
    if(line[:5] == "MOTIF"):
        if(outputFile): outputFile.close()
        outputFile = open(outputLocation+line.strip().split(" ")[1]+".meme","w")
        outputFile.write("MEME version 4.4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        outputFile.write("Background letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")
    if(outputFile): outputFile.write(line)

# Termination
outputFile.close()
#os.system("rm "+memeFileName)


