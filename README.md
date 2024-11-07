# Unsupervised Segmentation Pipeline

## Step 1: Prepare Image Dataset
1. **Select Images**: Randomly select 120 images from your cohort.
2. **Create CSV**: Create a CSV file containing the names of the 120 selected images. The CSV should have a column labeled `slide_id` and look like this: [ref](./samples/slides_list.csv)

   | slide_id     |
   |--------------|
   | example_001.jpg  |
   | example_002.jpg  |
   | example_003.jpg  |
   | ...          |

## Step 2: Generate Segmentation Maps

- Use the [main](./code/main.py) script to plot segmentation maps. The script requires the following arguments:
  - **images_folder**: Directory where the slide images are located.
  - **patch_size**: Size of the patch to be extracted.
  - **hdf5_folder**: Directory to save the extracted patches and coordinates.
  - **csv_path**: Path to the CSV file containing the 120 slide IDs.
  - **feat_dir**: Directory for saving the extracted features.
  - **n_comp**: Number of components for PCA.
  - **clust_dir**: Directory to save the results from clustering (generated labels).
  - **n_clust**: Number of desired clusters.
  - **mask_folder**: Directory for saving created masks.

- The segmentations produced can then be used to run the *Normalization Pipeline*.

---

# Normalization Pipeline

## Step 1: Prepare Image Dataset with Masks
1. **Select Images and Masks**: Obtain 120 randomly selected images and their corresponding multi-label segmentation masks.
2. **Organize Folders**: Place the images and masks into separate folders.

## Step 2: Run the Normalization Pipeline

- The normalization pipeline consists of three main scripts:
  1. [Produce_Image_Stain_Vectors_and_Optical_Density](./code/src/1-Produce_Image_Stain_Vectors_and_Optical_Density.py)
  2. [Aggregate_Stain_Vectors_and_Histograms](./code/src/2-Aggregate_Stain_Vectors_and_Histograms.py)
  3. [Normalize_Image](./code/src/3-Normalize_Image.py)

- Use the [run](./code/src/run.sh) script to execute the full pipeline. Update the paths for the following variables in the script:
  - **Image_Array**: List of paths to the 120 images used for producing stain vectors.
  - **Image_Map_Array**: List of paths to the 120 segmentation masks corresponding to the 120 images.
  - **All_Images_Array**: List of paths to the full dataset (images to be normalized).
  - **Excluding_Labels**: Any labels or regions to exclude.
  - **Python_Scripts_Directory**: Path to the directory containing all Python scripts.
  - **Images_Directory**: Path to the directory with 120 images for normalization.
  - **Image_Maps_Directory**: Path to the directory with the corresponding 120 image masks.
  - **Gray_Level_Labels_Directory**: Path to a CSV file containing the following 3 columns: [ref](./samples/LV_Gray_Level_to_Label.csv))
    - **GrayLevel**: Range representing different ROIs (e.g., 0, 1, 2 for three regions).
    - **RgbColor**: RGB values representing pixel intensity for each ROI.
    - **FeatureLabel**: Descriptive label for each ROI.
  
  - **Output_Files**: Path to the directory for saving all intermediate and final outputs:
    - Outputs from script 1:
      - `Images_Histograms_DataFrames`
      - `Images_Stain_Stats_DataFrames`
    - Outputs from script 2:
      - `Normalization_Parameters`
    - Output from script 3:
      - `Normalized Images`

## File Format and Naming Conventions
- **Image Files**: Can be in `.jpg`, or `.png` formats (non-pyramidal).
- **Image Map Files**: Should be in `.jpg` or `.png` format.
- **Naming Convention**: Image map filenames must include the corresponding image name. For example, if the image is `292324400.jpg`, the map should be named something like `LM_292324400.png`.
  - To locate the map, the script will search for `292324400` within the map file name.
  - Similarly, the image name can be derived by splitting the name of the map file by “_” and then by “.” to match it to the corresponding image.

