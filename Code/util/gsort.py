#################################################################################################
# Contains sort operations.
#################################################################################################

# Import
import operator

def sortBedDictionaries(dictList, field=0):
    """ 
    Sorts each dictionary of a given vector of dictionaries by a specified bed field.
      dictList: List of dictionaries to sort.
      field: Number of field in which the sort will be based. Start (and default) from 0 (pos1 field).
    """
    newList = []
    for d in dictList:
        newDict = dict()
        for c in d.keys():
            newDict[c] = sorted(d[c],key=operator.itemgetter(field))
        newList.append(newDict)
    return newList
# End sortBedDictionaries

def sortBedList(coordList, field=0):
    """ 
    Sorts a list of coordinates in bed format.
      coordList: List of coordinates to sort.
      field: Number of field in which the sort will be based.
    """
    return sorted(coordList,key=operator.itemgetter(field))
# End sortBedDictionaries


