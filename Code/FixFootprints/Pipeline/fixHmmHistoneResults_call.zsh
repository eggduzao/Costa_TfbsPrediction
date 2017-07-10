#!/bin/zsh

###################################################################################################
# A. Operations:
#    1. Convert all bed results to bigbed
#    2. Extract footprints
#    3. Cut HMM predictions by DNase+histone enriched regions (peaks)
#    4. Create extended footprints without merging
###################################################################################################

# Parameters
#il="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"
#inputList=( $il*"/H_HMM/"*".bed" )

# Execution
#for inputFile in $inputList
#do
    #bsub -J "FHR" -o "FHR_out.txt" -e "FHR_err.txt" -W 5:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" 
#    ./fixHmmHistoneResults_pipeline.zsh $inputFile "1"
#done

###################################################################################################
# B. Operations:
#    1. Merge different histone HMM results
###################################################################################################

# Cell
cellList=( "H1hesc" "K562" )
for cell in $cellList
do

  # HMM Cell Model
  hmmList=( "H1hesc" "K562" )
  for hmm in $hmmList
  do

    # Combinations (Double)
    inList1=( "H3K4me1" "H3K4me3" )
    inList2=( "H3K4me3" "H3K9ac" )
    outList=( "D13" "D39" )

    # Combinations (Triple)
    #inList1=( "H3K4me1" )
    #inList2=( "H3K4me3" )
    #inList3=( "H3K9ac" )
    #outList=( "D139" )

    # Execution
    for i in {1..$#inList1}
    do

      # Fetching files
      inLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"
      fileName1=$inLoc$cell"/HMMH_"$inList1[$i]"_Model"$hmm
      fileName2=$inLoc$cell"/HMMH_"$inList2[$i]"_Model"$hmm
      #fileName3=$inLoc$cell"/HMMH_"$inList3[$i]"_Model"$hmm
      outputFile=$inLoc$cell"/HMMH_"$outList[$i]"_Model"$hmm

      # B.1.1. Merging files
      #cat $fileName1".bed" $fileName2".bed" $fileName3".bed" > $outputFile"_Tmerge.bed"
      cat $fileName1".bed" $fileName2".bed" > $outputFile"_Tmerge.bed"

      # B.1.2. Sorting files for mergeBed
      sort -k1,1 -k2,2n $outputFile"_Tmerge.bed" > $outputFile"_Tsort.bed"

      # B.1.3. Performing mergeBed
      mergeBed -i $outputFile"_Tsort.bed" > $outputFile".bed"

      # Removing temporary files
      rm $outputFile"_Tmerge.bed" $outputFile"_Tsort.bed"

      # C.1. Create extended footprints without merging
      ./fixHmmHistoneResults_pipeline.zsh $outputFile".bed" "2"

    done
  done
done


