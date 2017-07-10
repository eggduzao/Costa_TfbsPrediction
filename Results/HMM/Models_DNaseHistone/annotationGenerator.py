import sys
i1 = int(sys.argv[1])
i2 = int(sys.argv[2])
outFileName = sys.argv[3]
offset = 20
outFile = open(outFileName,"w")
outFile.write("\n# Coordinates are 1-relative (Genome browser and wig files)\n0 = BACK\n1 = UPH\n2 = TOPH\n3 = DOWNH\n4 = UPD\n5 = TOPD\n6 = DOWND\n7 = FP\n\n")
counter = 0
for i in range(i1,i2+1,offset):
    outFile.write(str(i)[-5:]+" 00000 00000 00000 00000 "+str(i+offset-1)[-5:]+"\n")
    counter += 1
    if(counter == 5):
        outFile.write("\n")
        counter = 0
outFile.close()

