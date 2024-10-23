#!/usr/bin/env bash
set -ex

# Images, image maps, tissue to exclude, and output dataframe name
Image_Array=($(ls ./data/for_normalization/Images))
Image_Map_Array=($(ls ./data/for_normalization/KM_Masks/))
All_Images_Array=($(ls ./data/Images))
Excluding_Labels=("") # if any

# File Paths
Python_Scripts_Directory="./code/src/"
Images_Directory="./data/for_normalization/Images/"
All_Images_Directory="./data/Images/"
Image_Maps_Directory="./data/for_normalization/KM_Masks/"
Gray_Level_Labels_Directory="./data/Csv_Files/"
Output_Files="./results/clustering/"

# Other variables
Training_Time=10
Knuth_Bin_Size=4096

# Sorting function
custom_sort() {
    local IFS='_'
    read -ra parts1 <<< "$1"
    read -ra parts2 <<< "$2"
    number1="${parts1[-1]}"
    number2="${parts2[-1]}"
    if (( number1 < number2 )); then
        echo "-1"
    elif (( number1 > number2 )); then
        echo "1"
    else
        echo "0"
    fi
}

# Sort Image_Map_Array based on the slide number
sorted_Image_Map_Array=($(printf "%s\n" "${Image_Map_Array[@]}" | sort -t '_' -k 4,4n -k 5,5n -k 6,6n -k 7,7n))

# Verify the sorted array
Image_Map_Array=("${sorted_Image_Map_Array[@]}")
# Excluding_Labels=("")
Output_Dataframe_Name=()
for image in ${Image_Array[@]}; do
    filename=$(basename "$image")
    filename_without_extension="${filename%.*}"
    Output_Dataframe_Name+=("Dataframe_${filename_without_extension}")
done

# Create Directories
mkdir -p ./results/clustering/Images_Histograms_DataFrames
mkdir -p ./results/clustering/Images_Stain_Stats_DataFrames
mkdir -p ./results/clustering/Normalization_Parameters
mkdir -p ./results/clustering/Normalized_Images
mkdir -p ./results/clustering/Normalization_Parameters/${#Image_Array[@]}_Image_Cohort_Aggregated_Normalization_Parameters

# 1) Calculate stain vectors and histogram for each image and store info in a dataframe

for i in ${!Image_Array[@]}; do
    echo "----------------------------------------------------"
    echo "Generate pandas dataframes containing stain vectors "
    echo "and optical density for each cohort image           "
    echo "----------------------------------------------------"
    python $Python_Scripts_Directory"1-Produce_Image_Stain_Vectors_and_Optical_Density.py" \
    --Slide_Image                $Images_Directory${Image_Array[$i]} \
    --Label_Map_Image            $Image_Maps_Directory${Image_Map_Array[$i]} \
    --Gray_Level_To_Label_Legend $Gray_Level_Labels_Directory"New_Gray_Level_to_Label.csv" \
    --Output_Dataframe_File      $Output_Files${Output_Dataframe_Name[$i]} \
    --Excluding_Labels           "${Excluding_Labels[0]}" \
    --Bin_Size                   $Knuth_Bin_Size \
    --Stain_Vector_Training      $Training_Time
done

# 2) Aggregate stain vectors and histogram from the images in step 1

echo "----------------------------------------------------"
echo "Aggregate stain vectors and histograms              "
echo "----------------------------------------------------"
python $Python_Scripts_Directory"2-Aggregate_Stain_Vectors_and_Histograms.py" \
--Histogram_Dataframe_Directory    $Output_Files"Images_Histograms_DataFrames" \
--Stain_Vector_Dataframe_Directory $Output_Files"Images_Stain_Stats_DataFrames" \
--Output_Directory                 $Output_Files"Normalization_Parameters" \
--Number_of_Images                 ${#Image_Array[@]}

# 3) Normalize each image using aggregated stain vectors and histogram

for i in ${!All_Images_Array[@]}; do
    echo "----------------------------------------------------"
    echo "Normalize image using aggregated parameters         "
    echo "----------------------------------------------------"
    python $Python_Scripts_Directory"3-Normalize_Image.py"\
    --Image_To_Normalize         $All_Images_Directory${All_Images_Array[$i]} \
    --Normalizing_Histogram      $Output_Files"Normalization_Parameters/"${#Image_Array[@]}_Image_Cohort_Aggregated_Normalization_Parameters/${#Image_Array[@]}ImageCohortHistograms.npy \
    --Normalizing_Stain_Vectors  $Output_Files"Normalization_Parameters/"${#Image_Array[@]}_Image_Cohort_Aggregated_Normalization_Parameters/${#Image_Array[@]}ImageCohortStainVectors.npy \
    --Output_Directory           $Output_Files"Normalized_Images" \
    --Stain_Vector_Training      $Training_Time
done