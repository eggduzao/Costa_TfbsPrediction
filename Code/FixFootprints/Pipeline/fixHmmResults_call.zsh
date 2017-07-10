#!/bin/zsh

###################################################################################################
# A. Operations:
#    1. Convert all bed results to bigbed
#    2. Extract footprints
#    3. Cut HMM predictions by DNase+histone enriched regions (peaks)
#    4. Create extended footprints without merging
###################################################################################################

# Parameters
il="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"

#inputList_H_HD=( $il"H1hesc/DH_HMM_HD/"*".bed" )
#inputList_H_LOC=( $il"H1hesc/DH_HMM_LOC/"*".bed" )
#inputList_HE_HD=( $il"HeLaS3/DH_HMM_HD/"*".bed" )
#inputList_HP_HD=( $il"HepG2/DH_HMM_HD/"*".bed" )
#inputList_HU_HD=( $il"Huvec/DH_HMM_HD/"*".bed" )
#inputList_K_HD=( $il"K562/DH_HMM_HD/"*".bed" )
#inputList_K_LOC=( $il"K562/DH_HMM_LOC/"*".bed" )
#inputList_H_HD=( $il"H1hesc/DH_HMM_HD/H3K4me"*"_ModelHe"*".bed" )
#inputList_K_HD=( $il"K562/DH_HMM_HD/H3K4me"*"_ModelHe"*".bed" )

#inputList_H_DU_BIAS=( $il"H1hesc/HMM_DU_BIAS/"*".bed" )
#inputList_HE_DU_BIAS=( $il"HeLaS3/HMM_DU_BIAS/H3K4me1_ModelHeLaS3.bed" )
#inputList_HP_DU_BIAS=( $il"HepG2/HMM_DU_BIAS/"*".bed" )
#inputList_HU_DU_BIAS=( $il"Huvec/HMM_DU_BIAS/"*".bed" )
#inputList_K_DU_BIAS=( $il"K562/HMM_DU_BIAS/"*".bed" )

#inputList_HE_UW=( $il"HepG2/HMM_UW/"*".bed" )
#inputList_HU_UW=( $il"Huvec/HMM_UW/"*".bed" )
#inputList_K_UW=( $il"K562/HMM_UW/"*".bed" )

#inputList_HP_UW_BIAS=( $il"HepG2/HMM_UW_BIAS/"*".bed" )
#inputList_HU_UW_BIAS=( $il"Huvec/HMM_UW_BIAS/"*".bed" )
#inputList_K_UW_BIAS=( $il"K562/HMM_UW_BIAS/"*".bed" )

#inputList_DNaseOnly=( $il*/HMM_DNase_UW/*.bed )
#inputList_DNaseOnly_BIAS=( $il*/HMM_DNase_UW_BIAS/*.bed )

inputList=( 
$il"GM12878/HMM_DNase_DU_NAKED/"*.bed
)

# Execution
for inputFile in $inputList
do
  fixHmmResults $inputFile "1"
done

###################################################################################################
# B. Operations:
#    1. Merge different histone HMM results
###################################################################################################

# Cell Parameters
#cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )

# Cell Loop
#for cell in $cellList
#do

  # Type Parameters
#  if [[ $cell == "H1hesc" ]]; then
#    typeList=( "HMM_DU_BIAS" )
#  elif [[ $cell == "HeLaS3" ]]; then
#    typeList=( "HMM_DU_BIAS" )
#  elif [[ $cell == "HepG2" ]]; then
#    typeList=( "HMM_DU_BIAS" "HMM_UW_BIAS" "HMM_UW" )
#  elif [[ $cell == "Huvec" ]]; then
#    typeList=( "HMM_DU_BIAS" "HMM_UW_BIAS" "HMM_UW" )
#  elif [[ $cell == "K562" ]]; then
#    typeList=( "HMM_DU_BIAS" "HMM_UW_BIAS" "HMM_UW" )
#  fi

  # Footprint type loop
#  for type in $typeList
#  do

    # Parameters
#    inLoc="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"
#    fileName1=$inLoc$cell"/"$type"_H3K4me1_Model"$cell
#    fileName2=$inLoc$cell"/"$type"_H3K4me3_Model"$cell
#    outputFile=$inLoc$cell"/"$type"_D13_Model"$cell

    # B.1.1. Merging files
#    cat $fileName1".bed" $fileName2".bed" > $outputFile"_Tmerge.bed"

    # B.1.2. Sorting files for mergeBed
#    sort -k1,1 -k2,2n $outputFile"_Tmerge.bed" > $outputFile"_Tsort.bed"

    # B.1.3. Performing mergeBed
#    mergeBed -i $outputFile"_Tsort.bed" > $outputFile".bed"

    # Removing temporary files
#    rm $outputFile"_Tmerge.bed" $outputFile"_Tsort.bed"

#  done
#done


################################################################################################

# Cell
#cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
#cellList=( "H1hesc" "K562" )
#for cell in $cellList
#do

  # Windowing
  #winList=( "HD" "LOC" )
#  winList=( "LOC" )
#  for win in $winList
#  do

    # HMM Structure
    #strucList=( "_" "_com_" "_ext_" )
#    strucList=( "_" )
#    for struc in $strucList
#    do

      # HMM Cell Model
      #hmmList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )
#      hmmList=( "H1hesc" "K562" )
#      for hmm in $hmmList
#      do

        # Combinations (Double)
        #inList1=( "H2AZ" "H2AZ" "H2AZ" "H2AZ" "H3K4me1" "H3K4me1" "H3K4me1" "H3K4me3" "H3K4me3" "H3K9ac" )
        #inList2=( "H3K4me1" "H3K4me3" "H3K9ac" "H3K27ac" "H3K4me3" "H3K9ac" "H3K27ac" "H3K9ac" "H3K27ac" "H3K27ac" )
        #outList=( "DZ1" "DZ3" "DZ9" "DZ7" "D13" "D19" "D17" "D39" "D37" "D97" )

        # Combinations (Triple)
#        inList1=( "H2AZ"    "H2AZ"    "H2AZ"    "H2AZ"    "H2AZ"    "H2AZ"    "H3K4me1" "H3K4me1" "H3K4me1" "H3K4me3" )
#        inList2=( "H3K4me1" "H3K4me1" "H3K4me1" "H3K4me3" "H3K4me3" "H3K9ac"  "H3K4me3" "H3K4me3" "H3K9ac"  "H3K9ac" )
#        inList3=( "H3K4me3" "H3K9ac"  "H3K27ac" "H3K9ac"  "H3K27ac" "H3K27ac" "H3K9ac"  "H3K27ac" "H3K27ac" "H3K27ac" )
#        outList=( "DZ13"    "DZ19"    "DZ17"    "DZ39"    "DZ37"    "DZ97"    "D139"    "D137"    "D197"    "D397" )

        # Execution
#        for i in {1..$#inList1}
#        do

          # Fetching files
#          inLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"
#          fileName1=$inLoc$cell"/"$win"_"$inList1[$i]$struc"Model"$hmm
#          fileName2=$inLoc$cell"/"$win"_"$inList2[$i]$struc"Model"$hmm
#          fileName3=$inLoc$cell"/"$win"_"$inList3[$i]$struc"Model"$hmm
#          outputFile=$inLoc$cell"/"$win"_"$outList[$i]$struc"Model"$hmm

          # B.1.1. Merging files
#          cat $fileName1".bed" $fileName2".bed" $fileName3".bed" > $outputFile"_Tmerge.bed"

          # B.1.2. Sorting files for mergeBed
#          sort -k1,1 -k2,2n $outputFile"_Tmerge.bed" > $outputFile"_Tsort.bed"

          # B.1.3. Performing mergeBed
#          mergeBed -i $outputFile"_Tsort.bed" > $outputFile".bed"

          # Removing temporary files
#          rm $outputFile"_Tmerge.bed" $outputFile"_Tsort.bed"

          # C.1. Create extended footprints without merging
#          ./fixHmmResults_pipeline.zsh $outputFile".bed" "2"

#        done
#      done  
#    done
#  done
#done


