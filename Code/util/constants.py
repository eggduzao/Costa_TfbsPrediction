#################################################################################################
# Contains useful constants.
#################################################################################################

# Import
import os

def getChromSizesLocation():
    """ 
    Returns the location + name of the chrom sizes file for hg19.
    """
    return "/".join(os.path.realpath(__file__).split("/")[:-3]) + "/data/hg19.chrom.sizes"
    #return "/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/script/chrom.sizes"
# End getChromSizesLocation

def getChromList(x=True, y=True, reference=[]):
    """ 
    Returns the chromossome list
      x: Wether the chrX will be present or not.
      y: Wether the chrY will be present or not.
      reference: List of dictionaries. The returned chromList will only contain entries that appear on any of these dictionaries.
    """
    if(not reference):
        chromList = ["chr"+str(e) for e in range(1,23)]
        if(x): chromList.append("chrX")
        if(y): chromList.append("chrY")
    else:
        chromList = []
        for n in [str(e) for e in range(1,23)] + ["X","Y"]:
            appears = False
            for d in reference:
                if("chr"+n in d.keys()):
                    appears = True
                    break
            if(appears): chromList.append("chr"+n)
    return chromList
# End getChromList


