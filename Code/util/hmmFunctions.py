#################################################################################################
# Contains HMM methods
#################################################################################################

# Import
import aux
#from ghmm import *

def createHMM(inputFileName,returnMode="hmm"):
    """ 
    Creates an HMM based on .hmm file
      inputFileName: Input file (.hmm).
    """
    inputFile = open(inputFileName,"r")
    #emissionDomain = Float()
    hmmStates = int(inputFile.readline().strip().split(" ")[1])
    inputFile.readline()
    pi = [float(e) for e in inputFile.readline().strip().split(" ")]
    inputFile.readline()
    A = []
    for i in range(0,hmmStates): A.append([float(e) for e in inputFile.readline().strip().split(" ")])
    inputFile.readline()
    firstEmisLine = inputFile.readline()
    if("#" in firstEmisLine): # The HMM is multivariate
        E = []
        for i in range(0,hmmStates): 
            if(i==0): Elist = firstEmisLine.strip().split("#")
            else: Elist = inputFile.readline().strip().split("#")
            E.append([[float(e) for e in Elist[0].strip().split(" ")],[float(e) for e in Elist[1].strip().split(" ")]])
        dimNo = len(E[0][0])
        #hmm = HMMFromMatrices(emissionDomain,MultivariateGaussianDistribution(emissionDomain), A, E, pi)
    else: # The HMM is one-dimensional
        dimNo = 1
        E = []
        for i in range(0,hmmStates):
            if(i==0): Elist = firstEmisLine
            else: Elist = inputFile.readline()
            E.append([float(e) for e in Elist.strip().split(" ")])
        #hmm = HMMFromMatrices(emissionDomain,GaussianDistribution(emissionDomain), A, E, pi)
    inputFile.close()
    if(returnMode == "hmm"): return hmm, hmmStates, dimNo, emissionDomain
    elif(returnMode == "mat"): return hmmStates, dimNo, pi, A, E
    elif(returnMode == "sci"): 
        meansVec = [e[0] for e in E]
        stdVec = []
        for e in E:
            newMat = []
            for i in range(0,dimNo):
                newVec = []
                for j in range(0,dimNo):
                    newVec.append(e[1][(dimNo*i)+j])
                newMat.append(newVec)
            stdVec.append(newMat)
        return hmmStates, dimNo, pi, A, meansVec, stdVec
    else: return "Choose a correct return mode"
# End createDiscreteHMM


def writeHMM(pi, A, E, outputFileName, precision=4):
    """ 
    Writes an HMM into an outputFile
      pi, A, E = initial, transition and emission probabilities, respectively.
      outputFileName: Output file name (.hmm).
    """
    outputFile = open(outputFileName,"w")
    outputFile.write("states "+str(len(A))+"\n")
    outputFile.write("initial\n")
    outputFile.write(str(pi[0]))
    for e in pi[1:]: outputFile.write(" "+str(round(e,precision)))
    outputFile.write("\ntransitions\n")
    for e in A:
        outputFile.write(str(round(e[0],precision)))
        for v in e[1:]: outputFile.write(" "+str(round(v,precision)))
        outputFile.write("\n")    
    outputFile.write("emissions\n")
    if(not isinstance(E[0][0],list)):
        for e in E: outputFile.write(str(round(e[0],precision))+" "+str(round(e[1],precision))+"\n")
    else:
        for v in E: 
            for n in v[0]: outputFile.write(str(round(n,precision))+" ")
            outputFile.write("# "+str(round(v[1][0],precision)))
            for n in v[1][1:]:
                outputFile.write(" "+str(round(n,precision)))
            outputFile.write("\n")
    outputFile.close()
    return "ok"
# End writeHMM


