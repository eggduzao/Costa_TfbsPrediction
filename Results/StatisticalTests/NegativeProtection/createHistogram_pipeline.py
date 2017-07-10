
# Import
import os
import sys

# Input
label = sys.argv[1]
bedFileName1 = sys.argv[2]
bedFileName2 = sys.argv[3]
outFileName = sys.argv[4]
outFileNameSizeDist = sys.argv[5]
outFileNameStats = sys.argv[6]

# Parameters
ivanExt = 20
wsize = 100
to_remove = []

# Creating extended bed file
extBedFileName = bedFileName1+"_ext.bed"
ivanExtBedFileName = bedFileName1+"_iext.bed"
to_remove.append(extBedFileName)
to_remove.append(ivanExtBedFileName)
bedFile = open(bedFileName1,"r")
extBedFile = open(extBedFileName,"w")
ivanExtBedFile = open(ivanExtBedFileName,"w")
for line in bedFile:
  ll = line.strip().split("\t")
  m = (int(ll[1]) + int(ll[2]))/2
  extBedFile.write("\t".join([ll[0],str(m-(wsize/2)),str(m+(wsize/2))])+"\n")
  ivanExtBedFile.write("\t".join([ll[0],str(m-(ivanExt/2)),str(m+(ivanExt/2))])+"\n")
bedFile.close()
extBedFile.close()
ivanExtBedFile.close()

# Footprints that overlapped with PU1
newFpFileName = bedFileName2+"_temp.bed"
to_remove.append(newFpFileName)
os.system("intersectBed -wa -a "+bedFileName2+" -b "+ivanExtBedFileName+" | sort -k1,1 -k2,2n > "+newFpFileName)

# Creating intersect bed file
intBedFileName = bedFileName1+"_int.bed"
to_remove.append(intBedFileName)
os.system("intersectBed -wa -wb -a "+extBedFileName+" -b "+newFpFileName+" > "+intBedFileName)

# Evaluating statistics
uniqInterFileName = bedFileName1+"_int2.bed"
to_remove.append(uniqInterFileName)
os.system("intersectBed -wa -u -a "+extBedFileName+" -b "+newFpFileName+" > "+uniqInterFileName)
def file_len(fname):
  i = 0
  with open(fname) as f:
    for i, l in enumerate(f): pass
  return i + 1
npu = file_len(extBedFileName)
nint = file_len(uniqInterFileName)

# Creating histogram
intBedFile = open(intBedFileName,"r")
overlapVec = [0.0] * wsize
total = 0.0
for line in intBedFile:
  ll = line.strip().split("\t")
  a1 = int(ll[1])
  b1 = int(ll[4])
  b2 = int(ll[5])
  for i in range(max(b1-a1,0),min(b2-a1,wsize)): overlapVec[i] += 1.0
  total += 1.0
intBedFile.close()
overlapVec = [str(e/total) for e in overlapVec]

# Writing histogram
outFile = open(outFileName,"w")
outFile.write("\n".join([label]+overlapVec)+"\n")
outFile.close()

# Writing size distribution
outFile = open(outFileNameSizeDist,"w")
inFile = open(newFpFileName,"r")
outFile.write(label+"\n")
for line in inFile:
  ll = line.strip().split("\t")
  size = int(ll[2]) - int(ll[1])
  outFile.write(str(size)+"\n")
outFile.close()
inFile.close()

# Writing statistics
outFile = open(outFileNameStats,"w")
outFile.write("\n".join([label+"_PU1_with_FP(%)",str(round(float(nint)*100/npu,4))])+"\n")
outFile.close()

# Termination
for e in to_remove: os.system("rm "+e)


