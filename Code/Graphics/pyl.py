
def get_rate_dict(labelList):

  # Method -> [sbRate, myStep, bcType]
  rateDict = dict([

###################################################################################################
# DNASE
###################################################################################################

# HINT DNase Only

("HINT",[0.04,10,"B"]),
("HINT_UW",[0.04,10,"B"]),
("HINT_D_DU",[0.04,10,"B"]),
("HINT_D_UW",[0.04,10,"B"]),

("HINTBC",[0.04,10,"BC"]),
("HINTBC_UW",[0.04,10,"BC"]),
("HINT-BC_D_DU",[0.04,10,"BC"]),
("HINT-BC_D_UW",[0.04,10,"BC"]),

("HINTBCN",[0.04,10,"BCN"]),
("HINTBCN_UW",[0.04,10,"BCN"]),
("HINT-BCN_D_DU",[0.04,10,"BCN"]),
("HINT-BCN_D_UW",[0.04,10,"BCN"]),

# HINT H3K4me1+H3K4me3 Only

("HINT_H13_DU",[0.03,7,"B"]),

# HINT DNase+H3K4me1+H3K4me3

("HINT_D13_DU",[0.06,15,"B"]),
("HINT_D13_UW",[0.06,15,"B"]),

("HINT-BC_D13_DU",[0.06,20,"BC"]),
("HINT-BC_D13_UW",[0.06,20,"BC"]),

# Baseline

("FS_DU",[None,None,"B"]), # SC
("FS_UW",[None,None,"B"]), # SC
("Protection_DU",[None,None,"B"]), # SC
("TC_DU",[None,None,"B"]), # SC
("TCH_DU",[None,None,"B"]), # SC
("TC_UW",[None,None,"B"]), # SC
("PWM",[None,None,"B"]), # SC
("fdr_4",[None,None,"B"]), # SC

# Competing

("BinDNase",[0.022,5,"B"]),
("BinDNase_80",[0.022,5,"B"]),
("BinDNase_85",[0.020,6,"B"]),
("BinDNase_90",[0.025,3,"B"]),
("BinDNase_95",[0.028,3,"B"]),
("BinDNase_99",[0.020,4,"B"]),
("BinDNase_rank",[0.015,5,"B"]),

("Boyle",[0.02,5,"B"]),
("Boyle_DU",[0.02,5,"B"]),

("Centipede",[0.02,2,"B"]),
("Centipede_80",[0.008,1,"B"]),
("Centipede_85",[0.017,2,"B"]),
("Centipede_90",[0.02,2,"B"]),
("Centipede_95",[0.018,2,"B"]),
("Centipede_99",[0.015,2,"B"]),

("Cuellar",[None,None,"B"]), # SC
("Cuellar_80",[None,None,"B"]), # SC
("Cuellar_85",[None,None,"B"]), # SC
("Cuellar_90",[None,None,"B"]), # SC
("Cuellar_95",[None,None,"B"]), # SC
("Cuellar_99",[None,None,"B"]), # SC

("Dnase2Tf",[0.03,8,"B"]),
("Dnase2Tf_DU",[0.03,8,"B"]),
("Dnase2Tf_rank",[0.021,2,"B"]),

("FLR",[0.02,2,"B"]),
("FLR_80",[0.01,2,"B"]),
("FLR_85",[0.015,2,"B"]),
("FLR_90",[0.02,2,"B"]),
("FLR_95",[0.018,2,"B"]),
("FLR_99",[0.015,2,"B"]),

("Neph",[0.02,5,"B"]),
("Neph_DU",[0.02,5,"B"]),

("PIQ",[0.02,8,"B"]),
("PIQ_80",[0.015,2,"B"]),
("PIQ_85",[0.015,4,"B"]),
("PIQ_90",[0.02,8,"B"]),
("PIQ_95",[0.015,8,"B"]),
("PIQ_99",[0.01,8,"B"]),

("Wellington",[0.02,6,"B"]),
("Wellington_DU",[0.02,6,"B"]),
("Wellington_rank",[0.011,2,"B"]),

###################################################################################################
# ATAC
###################################################################################################

# HINT

("ATAC_HINT_oneside_1",[0.010,3,"B"]),
("ATAC_HINT_oneside_3",[0.02,3,"B"]),
("ATAC_HINT_oneside_5",[0.015,3,"B"]),
("ATAC_HINT_oneside_7",[0.012,3,"B"]),
("ATAC_HINT_twoside_1",[0.015,4,"B"]),
("ATAC_HINT_twoside_3",[0.025,4,"B"]),
("ATAC_HINT_twoside_5",[0.020,4,"B"]),
("ATAC_HINT_twoside_7",[0.020,4,"B"]),

("ATAC_HINT_twoside_1_SHIFT",[0.02,4,"B"]),
("ATAC_HINT_twoside_1_SHIFT_BC",[0.02,4,"BC"]),

("sc_ATAC_HINT_twoside_1",[0.010,3,"B"]),

("sc_ATAC_HINT_twoside_1_SHIFT",[0.015,3,"B"]),
("sc_ATAC_HINT_twoside_1_SHIFT_BC",[0.015,3,"BC"]),

# Baseline

("ATAC_TC_50",[None,None,"B"]), # SC
("scATAC_TC_50",[None,None,"B"]) # SC

  ])

  for e in labelList:

    if("TC" in e or "tc" in e or "Tc" in e or "FS" in e or "fs" in e or "Fs" in e or "PROTECTION" in e or "protection" in e or "Protection" in e or "PROT" in e or "Prot" in e or "prot" in e or "FOS" in e or "fos" in e or "Fos" in e):
      rateDict[e] = [None,None,"B"]
      continue

    bcs = "B"
    if("BC" in e or "bc" in e or "Bc" in e or "Bias" in e or "bias" in e or "BIAS" in e): bcs = "BC"
 
    pf = 0.005

    if("atac" in e or "ATAC" in e or "Atac" in e):

      if("sc" in e or "SC" in e or "single" in e or "SINGLE" in e or "Single" in e or "singlecell" in e or "SingleCell" in e or "Singlecell" in e or "SINGLECELL" in e or "single-cell" in e or "Single-Cell" in e or "Single-cell" in e or "SINGLE-CELL" in e or "single_cell" in e or "Single_Cell" in e or "Single_cell" in e or "SINGLE_CELL" in e or "Sc" in e):
  
        if("shift" in e or "SHIFT" in e or "Shift" in e): pf = 0.01
        else: pf = 0.008

      else:

        if("shift" in e or "SHIFT" in e or "Shift" in e): pf = 0.015
        else: pf = 0.012
    
    elif("dnase" in e or "DNase" in e or "DNAse" in e or "DNASE" in e or "Dnase" in e):

      if("sc" in e or "SC" in e or "single" in e or "SINGLE" in e or "Single" in e or "singlecell" in e or "SingleCell" in e or "Singlecell" in e or "SINGLECELL" in e or "single-cell" in e or "Single-Cell" in e or "Single-cell" in e or "SINGLE-CELL" in e or "single_cell" in e or "Single_Cell" in e or "Single_cell" in e or "SINGLE_CELL" in e or "Sc" in e):
  
        if("shift" in e or "SHIFT" in e or "Shift" in e): pf = 0.035
        else: pf = 0.035

      else:

        if("shift" in e or "SHIFT" in e or "Shift" in e): pf = 0.035
        else: pf = 0.035

    try: rateDict[e] = [((sum([float(ord(eee)) for eee in e]))%0.03)+pf,(sum([int(ord(eee)) for eee in e]))%11,bcs]
    except Exception: rateDict[e] = [0.01, 5, "B"]

  return rateDict


