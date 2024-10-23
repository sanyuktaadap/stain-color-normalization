# Name          : Produce_Image_Stain_Vectors_and_Optical_Density
# Description   : This script calculates the stain vectors and color
#                 histogram for H&E histology imageS

# Library imports.
from __future__     import division
from pathlib        import Path
from datetime       import datetime
from PIL            import Image
from skimage.io     import imread
import openslide
import re
import pathlib
import argparse
import Utilities
import numpy          as np
import pandas         as pd
import dask.array     as da
import pickle5 as pickle

__version__ = "0.0.1"

# Constants
Image.MAX_IMAGE_PIXELS      = None
HEMATOXYLIN_STAIN           = 0
EOSIN_STAIN                 = 1
OUTPUT_FILE_EXTENSION       = 'parquet'
STAIN_VECTORS_DATAFRAMES    = 'Images_Stain_Stats_DataFrames'
HISTOGRAMS_DATAFRAMES       = 'Images_Histograms_DataFrames'
HEMATOXYLIN_STAIN_LABEL     = 'Hematoxylin'
EOSIN_STAIN_LABEL           = 'Eosin'

# Global variables
GrayLevelLabelMapDataFrame  = pd.DataFrame()

# This class is initialized to store various statistical metrics related to the image
class CompositeStatistics:
    def __init__(self):
        self.SlideImageName                         = str()
        self.LabelMapImageName                      = str()
        self.FeatureName                            = str()

        self.HematoxylinAreaInPixels                = int()
        self.HematoxylinPixelDensity                = np.array([])

        self.HematoxylinStainOpticalDensity_Red     = int()
        self.HematoxylinStainOpticalDensity_Green   = int()
        self.HematoxylinStainOpticalDensity_Blue    = int()

        self.HematoxylinRGBStainVector_Red          = int()
        self.HematoxylinRGBStainVector_Green        = int()
        self.HematoxylinRGBStainVector_Blue         = int()

        self.HematoxylinDensityMeans                = float()
        self.HematoxylinDensitySDs                  = float()
        self.HematoxylinDensityMedians              = float()

        self.Hematoxylin_Image_Count                = int()
        self.Hematoxylin_Image_Mean                 = float()
        self.Hematoxylin_Image_Standard_deviation   = float()
        self.Hematoxylin_Image_Min                  = float()
        self.Hematoxylin_Image_25_Percent           = float()
        self.Hematoxylin_Image_50_Percent           = float()
        self.Hematoxylin_Image_75_Percent           = float()
        self.Hematoxylin_Image_Max                  = float()

        self.EosinAreaInPixels                      = int()
        self.EosinPixelDensity                      = np.array([])

        self.EosinStainOpticalDensity_Red           = int()
        self.EosinStainOpticalDensity_Green         = int()
        self.EosinStainOpticalDensity_Blue          = int()

        self.EosinRGBStainVector_Red                = int()
        self.EosinRGBStainVector_Green              = int()
        self.EosinRGBStainVector_Blue               = int()

        self.EosinDensityMeans                      = float()
        self.EosinDensitySDs                        = float()
        self.EosinDensityMedians                    = float()

        self.Eosin_Image_Count                      = int()
        self.Eosin_Image_Mean                       = float()
        self.Eosin_Image_Standard_deviation         = float()
        self.Eosin_Image_Min                        = float()
        self.Eosin_Image_25_Percent                 = float()
        self.Eosin_Image_50_Percent                 = float()
        self.Eosin_Image_75_Percent                 = float()
        self.Eosin_Image_Max                        = float()

    # Processes an image to calculate stain vectors, stain pixel density, and optical density
    # conversion from RGB to stain vectors for both Hematoxylin and Eosin
    def ImageComposite(self, ImageName, MapName, Feature, ImageData):
        global InputArguments
        RED_COLOR                               = 0
        GREEN_COLOR                             = 1
        BLUE_COLOR                              = 2

        SlideImageName                          = str()
        LabelMapImageName                       = str()

        HematoxylinAreaInPixels                 = int()
        HematoxylinPixelDensity                 = np.array([])

        HematoxylinStainOpticalDensity_Red      = int()
        HematoxylinStainOpticalDensity_Green    = int()
        HematoxylinStainOpticalDensity_Blue     = int()

        HematoxylinRGBStainVector_Red           = int()
        HematoxylinRGBStainVector_Green         = int()
        HematoxylinRGBStainVector_Blue          = int()

        HematoxylinDensityMeans                 = float()
        HematoxylinDensitySDs                   = float()
        HematoxylinDensityMedians               = float()

        # print('Define Image wide stats')
        Hematoxylin_Image_Count                 = int()
        Hematoxylin_Image_Mean                  = float()
        Hematoxylin_Image_Standard_Deviation    = float()
        Hematoxylin_Image_Min                   = float()
        Hematoxylin_Image_25_Percent            = float()
        Hematoxylin_Image_50_Percent            = float()
        Hematoxylin_Image_75_Percent            = float()
        Hematoxylin_Image_Max                   = float()

        # print('Define Eosin Info Variables')
        EosinAreaInPixels                       = int()
        EosinPixelDensity                       = np.array([])

        EosinStainOpticalDensity_Red            = int()
        EosinStainOpticalDensity_Green          = int()
        EosinStainOpticalDensity_Blue           = int()

        EosinRGBStainVector_Red                 = int()
        EosinRGBStainVector_Green               = int()
        EosinRGBStainVector_Blue                = int()

        EosinDensityMeans                       = float()
        EosinDensitySDs                         = float()
        EosinDensityMedians                     = float()

        # print('Define Image wide stats')
        Eosin_Image_Count                       = int()
        Eosin_Image_Mean                        = float()
        Eosin_Image_Standard_Deviation          = float()
        Eosin_Image_Min                         = float()
        Eosin_Image_25_Percent                  = float()
        Eosin_Image_50_Percent                  = float()
        Eosin_Image_75_Percent                  = float()
        Eosin_Image_Max                         = float()

        SlideImageName                              = Path(ImageName).name
        LabelMapImageName                           = Path(MapName).name
        print('Calculating Image Stain Vectors')
        StainVectors_S                              = Utilities.CalculateStainVector(ImageData,float(InputArguments.Stain_Vector_Lambda),int(InputArguments.Stain_Vector_Training))
        print('Calculating Image Density Map')
        StainPixelDensity_W                         = Utilities.CalculateDensityMap(ImageData, StainVectors_S,float(InputArguments.Density_Map_Lambda))
        print('Converting Stain Vectors from RGB to Optical Density')
        RgbStainVectors                             = Utilities.od2rgb(StainVectors_S)
        print('Extracting Hematoxylin Density')
        AllHematoxylinDensities                     = StainPixelDensity_W[:, HEMATOXYLIN_STAIN]
        print('Remove all pixels that do not actually contain hematoxylin')
        NonZeroHematoxylinDensities                 = AllHematoxylinDensities[np.nonzero(AllHematoxylinDensities)]

        HematoxylinPixelDensity                     = np.array(NonZeroHematoxylinDensities.tolist())
        HematoxylinAreaInPixels                     = len(NonZeroHematoxylinDensities)

        print('Hematoxylin Calculate Optical Density Stain Vectors')
        HematoxylinStainOpticalDensity_Red          = StainVectors_S[HEMATOXYLIN_STAIN][RED_COLOR]
        HematoxylinStainOpticalDensity_Green        = StainVectors_S[HEMATOXYLIN_STAIN][GREEN_COLOR]
        HematoxylinStainOpticalDensity_Blue         = StainVectors_S[HEMATOXYLIN_STAIN][BLUE_COLOR]

        print('Hematoxylin Calculate RGB Stain Vectors')
        HematoxylinRGBStainVector_Red               = RgbStainVectors[HEMATOXYLIN_STAIN][RED_COLOR]
        HematoxylinRGBStainVector_Green             = RgbStainVectors[HEMATOXYLIN_STAIN][GREEN_COLOR]
        HematoxylinRGBStainVector_Blue              = RgbStainVectors[HEMATOXYLIN_STAIN][BLUE_COLOR]

        print('Hematoxylin Density Stats')
        HematoxylinDensityMeans                     = np.mean(NonZeroHematoxylinDensities)
        HematoxylinDensitySDs                       = np.std(NonZeroHematoxylinDensities)
        HematoxylinDensityMedians                   = np.median(NonZeroHematoxylinDensities)

        print('Extracting Eosin Density')
        AllEosinDensities                           = StainPixelDensity_W[:, EOSIN_STAIN]
        print('Remove all pixels that do not actually contain Eosin')
        NonZeroEosinDensities                       = AllEosinDensities[np.nonzero(AllEosinDensities)]

        EosinPixelDensity                           = np.array(NonZeroEosinDensities.tolist())
        EosinAreaInPixels                           = len(NonZeroEosinDensities)

        print('Hematoxylin Calculate Optical Density Stain Vectors')
        EosinStainOpticalDensity_Red                = StainVectors_S[EOSIN_STAIN][RED_COLOR]
        EosinStainOpticalDensity_Green              = StainVectors_S[EOSIN_STAIN][GREEN_COLOR]
        EosinStainOpticalDensity_Blue               = StainVectors_S[EOSIN_STAIN][BLUE_COLOR]
        print('Eosin Calculate RGB Stain Vectors')
        EosinRGBStainVector_Red                     = RgbStainVectors[EOSIN_STAIN][RED_COLOR]
        EosinRGBStainVector_Green                   = RgbStainVectors[EOSIN_STAIN][GREEN_COLOR]
        EosinRGBStainVector_Blue                    = RgbStainVectors[EOSIN_STAIN][BLUE_COLOR]
        print('Eosin Density Stats')
        EosinDensityMeans                           = np.mean(NonZeroEosinDensities)
        EosinDensitySDs                             = np.std(NonZeroEosinDensities)
        EosinDensityMedians                         = np.median(NonZeroEosinDensities)
        print('------------------------------------')
        print('Hematoxylin Stats')
        print('\tDensity mean     : {}'.format(HematoxylinDensityMeans))
        print('\tDensity Std      : {}'.format(HematoxylinDensitySDs))
        print('\tDensity Median   : {}'.format(HematoxylinDensityMedians))
        print('\tNumber of pixels : {}'.format(HematoxylinAreaInPixels))
        print('\tStain Vectors    : {}'.format(RgbStainVectors[HEMATOXYLIN_STAIN]))
        print('------------------------------------')
        print('Eosin Stats')
        print('\tDensity mean     : {}'.format(EosinDensityMeans))
        print('\tDensity Std      : {}'.format(EosinDensitySDs))
        print('\tDensity Median   : {}'.format(EosinDensityMedians))
        print('\tNumber of pixels : {}'.format(EosinAreaInPixels))
        print('\tStain Vectors    : {}'.format(RgbStainVectors[EOSIN_STAIN]))
        print('------------------------------------')
        self.SlideImageName                         = SlideImageName
        self.LabelMapImageName                      = LabelMapImageName
        self.FeatureName                            = Feature

        self.HematoxylinAreaInPixels                = HematoxylinAreaInPixels
        self.HematoxylinStainOpticalDensity_Red     = HematoxylinStainOpticalDensity_Red
        self.HematoxylinStainOpticalDensity_Green   = HematoxylinStainOpticalDensity_Green
        self.HematoxylinStainOpticalDensity_Blue    = HematoxylinStainOpticalDensity_Blue

        self.HematoxylinRGBStainVector_Red          = HematoxylinRGBStainVector_Red
        self.HematoxylinRGBStainVector_Green        = HematoxylinRGBStainVector_Green
        self.HematoxylinRGBStainVector_Blue         = HematoxylinRGBStainVector_Blue

        self.HematoxylinDensityMeans                = HematoxylinDensityMeans
        self.HematoxylinDensitySDs                  = HematoxylinDensitySDs
        self.HematoxylinDensityMedians              = HematoxylinDensityMedians
        self.HematoxylinPixelDensity                = HematoxylinPixelDensity

        self.Hematoxylin_Image_Count                = Hematoxylin_Image_Count
        self.Hematoxylin_Image_Mean                 = Hematoxylin_Image_Mean
        self.Hematoxylin_Image_Standard_Deviation   = Hematoxylin_Image_Standard_Deviation
        self.Hematoxylin_Image_Min                  = Hematoxylin_Image_Min
        self.Hematoxylin_Image_25_Percent           = Hematoxylin_Image_25_Percent
        self.Hematoxylin_Image_50_Percent           = Hematoxylin_Image_50_Percent
        self.Hematoxylin_Image_75_Percent           = Hematoxylin_Image_75_Percent
        self.Hematoxylin_Image_Max                  = Hematoxylin_Image_Max

        self.EosinAreaInPixels                      = EosinAreaInPixels
        self.EosinStainOpticalDensity_Red           = EosinStainOpticalDensity_Red
        self.EosinStainOpticalDensity_Green         = EosinStainOpticalDensity_Green
        self.EosinStainOpticalDensity_Blue          = EosinStainOpticalDensity_Blue

        self.EosinRGBStainVector_Red                = EosinRGBStainVector_Red
        self.EosinRGBStainVector_Green              = EosinRGBStainVector_Green
        self.EosinRGBStainVector_Blue               = EosinRGBStainVector_Blue

        self.EosinDensityMeans                      = EosinDensityMeans
        self.EosinDensitySDs                        = EosinDensitySDs
        self.EosinDensityMedians                    = EosinDensityMedians
        self.EosinPixelDensity                      = EosinPixelDensity

        self.Eosin_Image_Count                      = Eosin_Image_Count
        self.Eosin_Image_Mean                       = Eosin_Image_Mean
        self.Eosin_Image_Standard_Deviation         = Eosin_Image_Standard_Deviation
        self.Eosin_Image_Min                        = Eosin_Image_Min
        self.Eosin_Image_25_Percent                 = Eosin_Image_25_Percent
        self.Eosin_Image_50_Percent                 = Eosin_Image_50_Percent
        self.Eosin_Image_75_Percent                 = Eosin_Image_75_Percent
        self.Eosin_Image_Max                        = Eosin_Image_Max

        print('Calculations completed')

# Function Name: Get Arguments
# Description: Define input arguments using flags
# Input: Slides, Label Map Color file, and output file
# Output: Argument list

def GetArguments():
    DESCRITPTION_MESSAGE = \
    'This scripts calculates the stain vectors and histogram for a given image.                       \n' + \
    'The calculation is based on Vahadane algorithm and loosely on work by manuscript titled:         \n' + \
    'Towards Population-based Histologic Stain Normalization of Glioblastoma. The script              \n' + \
    'calculates the parameters below per image and feature. Then, stores the results                  \n' + \
    'in a dataframe. The dataframe is saved in pickle format at given path for later                  \n' + \
    'analysis.                                                                                        \n' + \
    '                                                                                                 \n' + \
    'usage:                                                                                           \n' + \
    'Produce_Image_Stain_Vectors_and_Optical_Density.py                                               \n' + \
    '      --Slide_Image                266290664.jpg                                                 \n' + \
    '      --Label_Map_Image            W19-1-1-D.01_23_LM_266290664.png                              \n' + \
    '      --Gray_Level_To_Label_Legend LV_Gray_Level_to_Label.csv                                    \n' + \
    '      --Output_Dataframe_File      Dataframe_266290664                                           \n' + \
    '      --Excluding_Labels           \"\"                                                          \n'

    parser = argparse.ArgumentParser(description=DESCRITPTION_MESSAGE)
    # ------------------------------------
    parser.add_argument('-s', '--Slide_Image',                required=True,                    help='Slide Image')
    parser.add_argument('-l', '--Label_Map_Image',            required=True,                    help='Label Map Image')
    parser.add_argument('-g', '--Gray_Level_To_Label_Legend', required=True,                    help='CSV file containing gray level legend')
    parser.add_argument('-o', '--Output_Dataframe_File',      required=True,                    help='Output File where to place Dataframe results')
    parser.add_argument('-x', '--Excluding_Labels',           required=True,                    help='Feature Names to exclude. format Example: "Label 1, Label 2,...Label N"')
    parser.add_argument('-b', '--Bin_Size',                   required=False, default=32768,    help='Bin Size, use an integer')
    parser.add_argument('-t', '--Stain_Vector_Training',      required=False, default=240,      help='Stain Vector Training time in seconds')
    parser.add_argument('-e', '--Stain_Vector_Lambda',        required=False, default=0.1,      help='Stain Vector Lambda')
    parser.add_argument('-d', '--Density_Map_Lambda',         required=False, default=0.1,      help='Density Map Lambda')
    parser.add_argument('-v', '--version', action='version', version= "%(prog)s (ver: "+__version__+")")

    args = parser.parse_args()

    return args

# Function Name: Exclude Feature Labels
# Input: Excluding Feature List
# Output: gray level label

def ExcludeFeatureLabels(ExcludingFeatureList):
    GRAY_LEVEL_VALID_LABELS_TUPLE       = ('White/Background',
                                           'Spindled Patterns',
                                           'Pale and Yellow Cellularity',
                                           'High Cellular with Vacuoles',
                                           'Dense Cellular (Pronounced Vessels)',
                                           'Large Vessels (often Cauterized)',
                                           'Tissue Edges (Hemorrahic)',
                                           'Medium Cellularity'
                                           )
    EXCLUDING_LABELS_NAMES_REGEX        = '((?:\w+\s?){0,6}),?'
    FeatureName           = str()
    FeatureIndex          = int()
    GroupList             = list()
    NewGrayLevelLabelList = list(GRAY_LEVEL_VALID_LABELS_TUPLE)

    print('Check for empty input')
    if ExcludingFeatureList:
        print('Find all features to exclude')
        GroupList = re.findall(EXCLUDING_LABELS_NAMES_REGEX,ExcludingFeatureList)
        for FeatureName in GroupList:
            print(f'Excluding feature: {FeatureName}')
            if FeatureName in GRAY_LEVEL_VALID_LABELS_TUPLE:
                FeatureIndex = NewGrayLevelLabelList.index(FeatureName)
                NewGrayLevelLabelList.pop(FeatureIndex)
    else:
        print('No features to exclude')
        NewGrayLevelLabelList = GRAY_LEVEL_VALID_LABELS_TUPLE


    return NewGrayLevelLabelList

# Function Name: Creates a directory
# Description: Created a directory
# Input: path
# Output: output path

def CreateDirectory(OutputPath):
    try:
        Path(OutputPath).mkdir(parents=True, exist_ok=True)
    except:
        raise IOError()
    return str(OutputPath)

# Function Name: Import Gray Level Legend Data
# Description: Reads Label Map legend CSV data as a DataFrame and trims unused data
# Input: Data path
# Output: Dataframe

def ImportGrayLevelLegendData(SpreadsheetPath,GreyLevelLabels):
    # Initialize variables
    FEATURE_LABEL_COLUMN_TITLE    = 'FeatureLabel'
    GrayLevelLegendData           = pd.DataFrame()
    TrimmedGrayLevelLegendData    = pd.DataFrame()
    IndexedNewGrayLevelLegendData = pd.DataFrame()

    try:
        GrayLevelLegendData = pd.read_csv(SpreadsheetPath)
    except:
        print('No graylevel file data retrieved')
        raise IOError()

    print('Delete unwanted rows')
    TrimmedGrayLevelLegendData    = GrayLevelLegendData[GrayLevelLegendData.FeatureLabel.isin(GreyLevelLabels)]

    IndexedNewGrayLevelLegendData = TrimmedGrayLevelLegendData.set_index(FEATURE_LABEL_COLUMN_TITLE)

    return IndexedNewGrayLevelLegendData

# Function Name: Initialize
# Description: Sets up input, directories, and images for process
# Input: None
# Output: Image and map data

def Initialize():
    global GrayLevelLabelMapDataFrame
    global InputArguments

    print('----------------------------------------------------')
    print('Initialization Begins')
    print('Exclude invalid features')
    GreyLevelLabels                    = ExcludeFeatureLabels(InputArguments.Excluding_Labels)
    print('Create output directories')
    print('Import Gray Level Legend CSV file')
    GrayLevelLabelMapDataFrame         = ImportGrayLevelLegendData(InputArguments.Gray_Level_To_Label_Legend,GreyLevelLabels)

    DataFramePath   = pathlib.Path(str(Path(InputArguments.Output_Dataframe_File).parent / HISTOGRAMS_DATAFRAMES / Path(InputArguments.Output_Dataframe_File).name) + '.' + OUTPUT_FILE_EXTENSION)
    print('Test for File in Directory:{}'.format(DataFramePath))

    DataFramePath   = pathlib.Path(str(Path(InputArguments.Output_Dataframe_File).parent / STAIN_VECTORS_DATAFRAMES / Path(InputArguments.Output_Dataframe_File).name) + '.' + OUTPUT_FILE_EXTENSION)
    print('Test for File in Directory:{}'.format(DataFramePath))

    print('Get image pairs')
    SlideImageArray,\
    LabelMapImageArray      = LoadImagePairs(InputArguments.Slide_Image, InputArguments.Label_Map_Image)

    print('Initialization Ends')
    print('----------------------------------------------------')

    return SlideImageArray, LabelMapImageArray

# Function Name: Find and Label Unique Pixels
# Description: Match unique pixels to label map
# Input: Label map and image
# Output: Matching Features to gray level pixel map

def FindAndLabelUniquePixels(MapDataFrame, ImageLabelMap):  # Pixel by pixel find unique pixels
    global InputArguments
    GRAY_LEVEL_COLUMN_TITLE             = 'GrayLevel'
    FIRST_ITEM                          = 0

    print('Initialize variables')
    FeaturesFoundInImage = pd.DataFrame()

    print('Scanning for unique pixel colors in Label Map Legend')
    UniqueColors = np.unique(ImageLabelMap)

    for UniqueColor in UniqueColors:
        SeriesOfInterest        = MapDataFrame[GRAY_LEVEL_COLUMN_TITLE]
        BooleanSeriesOfInterest = SeriesOfInterest.isin([UniqueColor])
        FoundPixelInLabelMap    = MapDataFrame[BooleanSeriesOfInterest]

        if BooleanSeriesOfInterest.any():
            print('Found Grey Level pixel \"{}\" for feature: {}'.format(UniqueColor,FoundPixelInLabelMap.index[FIRST_ITEM]))
            FeaturesFoundInImage = pd.concat([FeaturesFoundInImage, FoundPixelInLabelMap])
    return FeaturesFoundInImage  # Annotations with matching pixels

# Function Name: Load Image Pairs
# Description: Loads image pairs
# Input: path and image pair names
# Output: Images

def LoadImagePairs(SlideName, LabelMapName):

    print('Processing Image Pairs: ')
    print('\tSlide: \t\t{}  '.format(Path(SlideName).name))
    print('\tLabel Map: \t{}'.format(Path(LabelMapName).name))

    TestFile(SlideName)
    TestFile(LabelMapName)

    # print('Loading Image: {}'.format(SlideName))
    # ImageOfInterestRGB = imread(SlideName)
    # print('Loading Image Map')
    # LabelMapImage      = imread(LabelMapName, as_gray=True)

    print('Loading Image: {}'.format(SlideName))
    if ".svs" in SlideName:
        slide = openslide.OpenSlide(SlideName)
        print("Slide opened")
        image_array = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]))
        print("converted into np array")
        dask_array = da.from_array(image_array)
        print("converted into dask array")
        ImageOfInterestRGB = dask_array[:, :, :3]
        print("RGB channels extracted")
    else:
        ImageOfInterestRGB = imread(SlideName)

    print('Loading Image Map')
    LabelMapImage      = imread(LabelMapName, as_gray=True)

    return ImageOfInterestRGB, LabelMapImage

# Function Name: Test file
# Description: Verify there are no problems with file
# Input: File name
# Output: None

def TestFile(FileName):
    try:

        FileNamePath = pathlib.Path(FileName)
        if not FileNamePath.exists():
            print('I/O error')
            print('File Name: {}'.format(Path(FileName).name))
            print('Path Name: {}'.format(Path(FileName).parent))
            raise IOError()
    except:
        print('Unexpected I/O error')
        raise IOError()

# Function Name: DataFrame to a Lists
# Description: Convert a dataframe to a list
# Input: Label Dataframe
# Output: Gray level, Feature, and RGB lists

def SplitLabelDataFrameToLists(DataFrame):
    RGB_COLOR       = -1
    GRAY_COLOR      = 0
    FeaturesList    = list()
    GrayLevelList   = list()
    RgbColorList    = list()

    GrayLevelDictionary = DataFrame.T.to_dict('list')
    for Features, GrayLevelValues in GrayLevelDictionary.items():
        GrayLevelList.append(GrayLevelValues[GRAY_COLOR])
        RgbColorList.append(GrayLevelValues[RGB_COLOR])
        FeaturesList.append(Features)

    return FeaturesList, GrayLevelList, RgbColorList

# Function Name: Build Mask for Target feature
# Description: Build Mask for Target feature
# Input: Image data, Gray Level values
# Output: Pixel Mask

def BuildMaskForTargetFeature(Image, GrayLevel):
    WHITE_COLOR = 255
    BLACK_COLOR = 0
    PixelMask = np.where(Image == GrayLevel, WHITE_COLOR, BLACK_COLOR)
    print('Pixel Mask Size               : {}'.format(PixelMask.shape))
    print('Pixel Mask based on Gray Level: {}'.format(GrayLevel))
    return PixelMask

# Function Name: Fetch Data Frame
# Description: Building list for dataframe export
# Input: Composite statistics data
# Output: List of Composite statistics data

def ClassDataToList(CompositeData):
    print('Initialize variables')

    HematoxylinComponentList = list()
    EosinComponentList       = list()

    print('Assign Hematoxylin data')
    HematoxylinComponentList = [CompositeData.SlideImageName,                       \
                                CompositeData.LabelMapImageName,                    \
                                CompositeData.FeatureName,                          \
                                HEMATOXYLIN_STAIN_LABEL,                            \
                                CompositeData.HematoxylinAreaInPixels,              \
                                CompositeData.HematoxylinPixelDensity,              \
                                CompositeData.HematoxylinStainOpticalDensity_Red,   \
                                CompositeData.HematoxylinStainOpticalDensity_Green, \
                                CompositeData.HematoxylinStainOpticalDensity_Blue,  \
                                CompositeData.HematoxylinRGBStainVector_Red,        \
                                CompositeData.HematoxylinRGBStainVector_Green,      \
                                CompositeData.HematoxylinRGBStainVector_Blue,       \
                                CompositeData.HematoxylinDensityMeans,              \
                                CompositeData.HematoxylinDensitySDs,                \
                                CompositeData.HematoxylinDensityMedians,            \
                                CompositeData.Hematoxylin_Image_Count,              \
                                CompositeData.Hematoxylin_Image_Mean,               \
                                CompositeData.Hematoxylin_Image_Standard_deviation, \
                                CompositeData.Hematoxylin_Image_Min,                \
                                CompositeData.Hematoxylin_Image_25_Percent,         \
                                CompositeData.Hematoxylin_Image_50_Percent,         \
                                CompositeData.Hematoxylin_Image_75_Percent,         \
                                CompositeData.Hematoxylin_Image_Max                 \
                                ]

    print('Assign Eosin data')
    EosinComponentList =       [CompositeData.SlideImageName,                       \
                                CompositeData.LabelMapImageName,                    \
                                CompositeData.FeatureName,                          \
                                EOSIN_STAIN_LABEL,                                  \
                                CompositeData.EosinAreaInPixels,                    \
                                CompositeData.EosinPixelDensity,                    \
                                CompositeData.EosinStainOpticalDensity_Red,         \
                                CompositeData.EosinStainOpticalDensity_Green,       \
                                CompositeData.EosinStainOpticalDensity_Blue,        \
                                CompositeData.EosinRGBStainVector_Red,              \
                                CompositeData.EosinRGBStainVector_Green,            \
                                CompositeData.EosinRGBStainVector_Blue,             \
                                CompositeData.EosinDensityMeans,                    \
                                CompositeData.EosinDensitySDs,                      \
                                CompositeData.EosinDensityMedians,                  \
                                CompositeData.Eosin_Image_Count,                    \
                                CompositeData.Eosin_Image_Mean,                     \
                                CompositeData.Eosin_Image_Standard_deviation,       \
                                CompositeData.Eosin_Image_Min,                      \
                                CompositeData.Eosin_Image_25_Percent,               \
                                CompositeData.Eosin_Image_50_Percent,               \
                                CompositeData.Eosin_Image_75_Percent,               \
                                CompositeData.Eosin_Image_Max                       \
                               ]
    print('Merge Hematoxylin & Eosin data')
    return [HematoxylinComponentList] + [EosinComponentList]

# Function Name: Get Image Composite Statistics
# Description:
# Input: none
# Output: none

def ExecuteDeconvolution(SlideImage,LabelMapImage):
    global InputArguments
    global GrayLevelLabelMapDataFrame
    NO_PIXELS                       = 0
    print('Initialize variables')
    MainList                        = list()
    FeatureList                     = list()
    GrayLevelList                   = list()
    RGBList                         = list()
    Feature                         = str()
    GrayLevel                       = int()
    PixelsFeaturesInImageDataFrame  = pd.DataFrame()
    Statistics                      = CompositeStatistics()

    print('Extract image file names')
    SlideImageName                 = str(Path(InputArguments.Slide_Image).name)
    LabelMapImageName              = str(Path(InputArguments.Label_Map_Image).name)

    print('List of colors present in the label map. Find all pixels with features')
    PixelsFeaturesInImageDataFrame = FindAndLabelUniquePixels(GrayLevelLabelMapDataFrame, LabelMapImage)

    print('Check for existing feature areas')
    if not PixelsFeaturesInImageDataFrame.empty:

        print('Fetch Gray Level Legend parameters')
        FeatureList, GrayLevelList, RGBList = SplitLabelDataFrameToLists(PixelsFeaturesInImageDataFrame)
        print('Scan through gray levels')
        for Feature, GrayLevel in zip(FeatureList, GrayLevelList):

            print('----------------------------------------')
            print('Feature: \"{}\", Gray level color: \"{}\"'.format(Feature, GrayLevel))
            print('----------------------------------------')

            Statistics = ApplyingFilters(SlideImage, SlideImageName, Statistics, LabelMapImage, LabelMapImageName,GrayLevel,Feature)

            print('If the feature has an area, keep feature statistics')
            HematoxylinPixelsFound      =  Statistics.HematoxylinAreaInPixels
            EosinPixelsFound            =  Statistics.EosinAreaInPixels

            if HematoxylinPixelsFound > NO_PIXELS or EosinPixelsFound > NO_PIXELS:
                print('Add feature statistics to output list')
                MainList               += ClassDataToList(Statistics)  # Concatenating lists is cheaper than dataframes
                print('Update feature counter')

            else:

                print('Discarded Feature: \"{}\", Gray level color: \"{}\"'.format(Feature, GrayLevel))
    else:
        print('Empty dataframe, no unique colors found')
        print('Dataframe will not hold any data')

    # To be safe, destroy class object
    del Statistics
    return MainList

# Function Name: Applying Filters
# Description: Discards small areas
# Input: SlideImage, SlideImageName, Statistics, LabelMapImage, LabelMapImageName, GrayLevel,Feature
# Output: Statistics

def ApplyingFilters(SlideImage, SlideImageName, Statistics, LabelMapImage, LabelMapImageName, GrayLevel,Feature):

    NO_PIXELS       = 0
    WHITE_COLOR     = 255
    PixelMask       = BuildMaskForTargetFeature(LabelMapImage, GrayLevel)
    print('Mask For Target Feature       : {}'.format(PixelMask.shape))
    SurvivingPixels  = np.count_nonzero(PixelMask)
    print('Surviving Pixels              : {}'.format(SurvivingPixels))

    print('Check for Areas to keep')
    if SurvivingPixels > NO_PIXELS:
        print('-----------------------------------------------')
        print('Processing Feature: {}'.format(Feature))
        print('WhiteMask: True where colored pixels, False where white pixels')
        WhiteMask   = PixelMask.astype(bool)

        print('Applying white mask to RGB image')
        NewImage    = np.where(WhiteMask[...,None], SlideImage, WHITE_COLOR)

        Statistics.ImageComposite(SlideImageName,LabelMapImageName,Feature,NewImage)

    else:
        print('-----------------------------------------------')
        print('No areas left to process after filters')
        print('Skipping feature: {}'.format(Feature))
        print('Output empty statistics class')
        Statistics  = CompositeStatistics()

    print('-----------------------------------------------')
    print('Calculating Statistics')
    print('Feature Name     : {}'.format(Feature))
    print('Surviving Pixels : {}'.format(SurvivingPixels))
    print('Image Name       : {}'.format(SlideImageName))
    print('Image Map Name   : {}'.format(LabelMapImageName))
    print('-----------------------------------------------')

    return Statistics

# Function Name: Terminate Process
# Description: Wraps up program
# Input: Areas Component List
# Output: Stain Vectors and Optical Density Dataframes

def Terminate(ComponentList):
    global InputArguments

    MAXIMUM_COLOR_BIN                   = 10
    MINIMUM_COLOR_BIN                   = 0
    PERCENT_MINIMUM_PIXEL_AREA          = 0.5  # 0.5% of WSI size
    DATAFRAME_COLUMN_NAMES              = {'SlideImageName'             :0,\
                                           'ImageLabelMapName'          :1,\
                                           'FeatureName'                :2,\
                                           'Stain'                      :3,\
                                           'Area'                       :4,\
                                           'PixelDensity'               :5,\
                                           'OpticalDensity_Red_S'       :6,\
                                           'OpticalDensity_Green_S'     :7,\
                                           'OpticalDensity_Blue_S'      :8,\
                                           'RGBStainVector_Red_W'       :9,\
                                           'RGBStainVector_Green_W'     :10,\
                                           'RGBStainVector_Blue_W'      :11,\
                                           'DensityMeans'               :12,\
                                           'StandardDev'                :13,\
                                           'MedianDensity'              :14,\
                                           'Image_Count'                :15,\
                                           'Image_Mean'                 :16,\
                                           'Image_Standard_Deviation'   :17,\
                                           'Image_Min'                  :18,\
                                           'Image_25_Percent'           :19,\
                                           'Image_50_Percent'           :20,\
                                           'Image_75_Percent'           :21,\
                                           'Image_Max'                  :22\
                                          }

    print('Initialize variables')
    BinsArray              = np.arange(MINIMUM_COLOR_BIN, MAXIMUM_COLOR_BIN, MAXIMUM_COLOR_BIN/int(InputArguments.Bin_Size))
    DataframeIndex         = np.arange(int(InputArguments.Bin_Size)-1).tolist()
    DataBase               = pd.DataFrame([], columns=[*DATAFRAME_COLUMN_NAMES])
    DaskBinsArray          = da.from_array(BinsArray)
    HematoxylinHistogram   = da.from_array([0]*len(DataframeIndex))
    EosinHistogram         = da.from_array([0]*len(DataframeIndex))
    DataFramePath          = str()
    HematoxylinList        = list()
    EosinList              = list()

    print('Check for missing feature information')
    if len(ComponentList):

        print('Fill Image Dataframe')
        DataBase           = pd.DataFrame(ComponentList, columns=[*DATAFRAME_COLUMN_NAMES])
        Stain_Name = HEMATOXYLIN_STAIN_LABEL

        print('Process {} Stain'.format(Stain_Name))
        Hematoxylin_Total_Pixel_Area   = DataBase[DataBase.Stain == Stain_Name].Area.sum()
        Hematoxylin_Area_Threshold     = int(PERCENT_MINIMUM_PIXEL_AREA * Hematoxylin_Total_Pixel_Area / 100)

        print('Image Pixel Area Threshold: {}% or {} pixels, for {} pixels total'.format(PERCENT_MINIMUM_PIXEL_AREA,Hematoxylin_Area_Threshold,Hematoxylin_Total_Pixel_Area))
        Filtered_Dataframe = DataBase[(DataBase.Area > Hematoxylin_Area_Threshold) & (DataBase.Stain == Stain_Name)]

        print('Check for feature with succifient area: {} of {} pixels'.format(Hematoxylin_Area_Threshold,Hematoxylin_Total_Pixel_Area))
        Number_Of_Survaving_Features = len(Filtered_Dataframe.index)

        if Number_Of_Survaving_Features:
            print('Number of Survaving Features: {}'.format(Number_Of_Survaving_Features))

            for Index_Row, Feature_Row in Filtered_Dataframe.iterrows():
                print('Exploring Feature: {}'.format(Feature_Row.FeatureName))
                if Feature_Row.Stain == Stain_Name:
                    HematoxylinList +=  Feature_Row.PixelDensity.tolist()

                    print('Delete Pixel Density from DataFrame')
                    DataBase.loc[Index_Row,'PixelDensity']= np.array([0])
        # --------------------------------------------
        Stain_Name = EOSIN_STAIN_LABEL
        print('Process {} Stain'.format(Stain_Name))
        Eosin_Total_Pixel_Area   = DataBase[DataBase.Stain == Stain_Name].Area.sum()
        Eosin_Area_Threshold     = int(PERCENT_MINIMUM_PIXEL_AREA * Eosin_Total_Pixel_Area / 100)
        print('Image Pixel Area Threshold: {}% or {} pixels, for {} pixels total'.format(PERCENT_MINIMUM_PIXEL_AREA,Eosin_Area_Threshold,Eosin_Total_Pixel_Area))
        Filtered_Dataframe = DataBase[(DataBase.Area > Eosin_Area_Threshold) & (DataBase.Stain == Stain_Name)]

        print('Check for feature with succifient area: {} of {} pixels'.format(Eosin_Area_Threshold,Eosin_Total_Pixel_Area))
        Number_Of_Survaving_Features = len(Filtered_Dataframe.index)
        if Number_Of_Survaving_Features:

            print('Number of Survaving Features: {}'.format(Number_Of_Survaving_Features))
            for Index_Row, Feature_Row in Filtered_Dataframe.iterrows():
                print('Exploring Feature: {}'.format(Feature_Row.FeatureName))

                if Feature_Row.Stain == Stain_Name:
                    EosinList +=  Feature_Row.PixelDensity.tolist()
                    print('Delete Pixel Density from DataFrame')

                    DataBase.loc[Index_Row,'PixelDensity']= np.array([0])

        if len(HematoxylinList):
            Hematoxylin_Statistics = pd.Series(HematoxylinList).describe()
            print('Calculating {} Histogram'.format('Hematoxylin'))
            HematoxylinHistogram,_ = da.histogram(da.from_array(HematoxylinList),DaskBinsArray)

        if len(EosinList):
            Eosin_Statistics       = pd.Series(EosinList).describe()
            print('Calculating {} Histogram'.format('Eosin'))
            EosinHistogram,_        = da.histogram(da.from_array(EosinList)      ,DaskBinsArray)

        print('Build Stain Dictionary')
        StainDictionary      = {'idx'                          :DataframeIndex,\
                                HEMATOXYLIN_STAIN_LABEL.lower():HematoxylinHistogram,\
                                EOSIN_STAIN_LABEL.lower()      :EosinHistogram}
        StainHistograms      = pd.DataFrame(StainDictionary)
        print('Create Index in dataframe')
        StainHistograms      = StainHistograms.set_index('idx')
        print('Type Cast Dataframe to int16')
        StainHistograms.astype(np.uint16).dtypes
        DataFramePath        = str(Path(InputArguments.Output_Dataframe_File).parent / HISTOGRAMS_DATAFRAMES / Path(InputArguments.Output_Dataframe_File).name) + '.' + OUTPUT_FILE_EXTENSION
        print('Create Stain Density Dask Dataframe')
        print('Save dataframe to disk')
        StainHistograms.to_parquet(DataFramePath,engine='pyarrow')

        if len(HematoxylinList):
            Stain_Name = 'Hematoxylin'
            Area_Threshold   = Hematoxylin_Area_Threshold
            Stain_Statistics = Hematoxylin_Statistics
            print('Insert Image Wide Statistics for {}'.format(Stain_Name))
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Count'             ] = Stain_Statistics['count']
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Mean'              ] = Stain_Statistics['mean' ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Standard_Deviation'] = Stain_Statistics['std'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Min'               ] = Stain_Statistics['min'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_25_Percent'        ] = Stain_Statistics['25%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_50_Percent'        ] = Stain_Statistics['50%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_75_Percent'        ] = Stain_Statistics['75%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Max'               ] = Stain_Statistics['max'  ]

        if len(EosinList):
            Stain_Name = 'Eosin'
            Area_Threshold   = Eosin_Area_Threshold
            Stain_Statistics = Eosin_Statistics
            print('Insert Image Wide Statistics for {}'.format(Stain_Name))
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Count'             ] = Stain_Statistics['count']
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Mean'              ] = Stain_Statistics['mean' ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Standard_Deviation'] = Stain_Statistics['std'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Min'               ] = Stain_Statistics['min'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_25_Percent'        ] = Stain_Statistics['25%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_50_Percent'        ] = Stain_Statistics['50%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_75_Percent'        ] = Stain_Statistics['75%'  ]
            DataBase.loc[(DataBase.Stain == Stain_Name) & (DataBase.Area > Area_Threshold),'Image_Max'               ] = Stain_Statistics['max'  ]

        print('Save dataframe to disk')
        DataFramePath             = str(Path(InputArguments.Output_Dataframe_File).parent / STAIN_VECTORS_DATAFRAMES / Path(InputArguments.Output_Dataframe_File).name) + '.' + OUTPUT_FILE_EXTENSION
        print('Concatenate both stains dataframes')
        Filtered_Dataframe = pd.concat([DataBase[(DataBase.Stain == HEMATOXYLIN_STAIN_LABEL) & (DataBase.Area > Hematoxylin_Area_Threshold)],\
                                        DataBase[(DataBase.Stain == EOSIN_STAIN_LABEL)       & (DataBase.Area > Eosin_Area_Threshold)]])
        Filtered_Dataframe['PixelDensity'] = Filtered_Dataframe['PixelDensity'].apply(lambda x: pickle.dumps(x))
        Filtered_Dataframe.to_parquet(DataFramePath,engine='pyarrow')

        Update_User= [['Image_Name'                   , Path(DataFramePath).name],                                                                                    \
                      ['Hematoxylin_Area_Threshold'    , Hematoxylin_Area_Threshold],                                                                                 \
                      ['Hematoxylin_Area_Total'        , Hematoxylin_Total_Pixel_Area],                                                                               \
                      ['Hematoxylin_Total_Features'    , len(DataBase[DataBase['Stain']==HEMATOXYLIN_STAIN_LABEL].index)],                                            \
                      ['Hematoxylin_Features_Kept'     , len(Filtered_Dataframe[Filtered_Dataframe['Stain']==HEMATOXYLIN_STAIN_LABEL].index)],                        \
                      ['Hematoxylin_Features_Disposed' , len(DataBase[(DataBase.Area <= Hematoxylin_Area_Threshold) & (DataBase['Stain']==HEMATOXYLIN_STAIN_LABEL)])],\
                      ['Eosin_Area_Threshold'          , Eosin_Area_Threshold],                                                                                       \
                      ['Eosin_Area_Total'              , Eosin_Total_Pixel_Area],                                                                                     \
                      ['Eosin_Total_Features'          , len(DataBase[DataBase['Stain']==EOSIN_STAIN_LABEL].index)],                                                  \
                      ['Eosin_Features_Kept'           , len(Filtered_Dataframe[Filtered_Dataframe['Stain']==EOSIN_STAIN_LABEL].index)],                              \
                      ['Eosin_Features_Disposed'       , len(DataBase[(DataBase.Area <= Eosin_Area_Threshold) & (DataBase['Stain']==EOSIN_STAIN_LABEL)])],            \
                      ['Stain_Vector_Training_Time'    , int(InputArguments.Stain_Vector_Training)],                                                                  \
                      ['Stain_Vector_Lambda'           , float(InputArguments.Stain_Vector_Lambda)],                                                                  \
                      ['Density_Map_Lambda'            , float(InputArguments.Density_Map_Lambda)],                                                                   \
                      ['Dataframe_Number_of_Columns'   , len(DataBase.columns)],                                                                                      \
                      ['Dataframe_Number_of_Rows'      , len(DataBase.index)],                                                                                        \
                      ['Dataframe_File_Path'           , DataFramePath]                                                                                               \
                      ]
        Info_Dataframe = pd.DataFrame(Update_User,columns=['Parameter','Value'])

        print('---------------------------------------------')
        print('Script Summary                   ')
        print('---------------------------------------------')
        print('Dataframe File Name              : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Image_Name'].Value.values[0]))
        print('---------------------------------------------')
        print('Stain Name: {}                           '.format(HEMATOXYLIN_STAIN_LABEL))
        print('Area Threshold Area/Total Area   : {}% or {} pixels : {} pixels total'.format(PERCENT_MINIMUM_PIXEL_AREA,\
                                                                 Info_Dataframe[Info_Dataframe.Parameter=='Hematoxylin_Area_Threshold'   ].Value.values[0],\
                                                                 Info_Dataframe[Info_Dataframe.Parameter=='Hematoxylin_Area_Total'       ].Value.values[0]))
        print('Total Number of Features Found   : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Hematoxylin_Total_Features'   ].Value.values[0]))
        print('Total Number of Features Kept    : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Hematoxylin_Features_Kept'    ].Value.values[0]))
        print('Total Number of Features Disposed: {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Hematoxylin_Features_Disposed'].Value.values[0]))
        print('---------------------------------------------')
        print('Stain Name: {}                           '.format(EOSIN_STAIN_LABEL))
        print('Area Threshold Area/Total Area   : {}% or {} pixels : {} pixels total'.format(PERCENT_MINIMUM_PIXEL_AREA,\
                                                                 Info_Dataframe[Info_Dataframe.Parameter=='Eosin_Area_Threshold'         ].Value.values[0],\
                                                                 Info_Dataframe[Info_Dataframe.Parameter=='Eosin_Area_Total'             ].Value.values[0]))
        print('Total Number of Features Found   : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Eosin_Total_Features'         ].Value.values[0]))
        print('Total Number of Features Kept    : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Eosin_Features_Kept'          ].Value.values[0]))
        print('Total Number of Features Disposed: {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Eosin_Features_Disposed'      ].Value.values[0]))
        print('---------------------------------------------')
        print('Stain Vector Training Time       : {} sec'.format(Info_Dataframe[Info_Dataframe.Parameter=='Stain_Vector_Training_Time'   ].Value.values[0]))
        print('Stain Vector Lambda              : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Stain_Vector_Lambda'          ].Value.values[0]))
        print('Density Map Lambda               : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Density_Map_Lambda'           ].Value.values[0]))
        print('---------------------------------------------')
        print('Dataframe Total Number of Columns: {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Dataframe_Number_of_Columns'  ].Value.values[0]))
        print('Dataframe Total Number of Rows   : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Dataframe_Number_of_Rows'     ].Value.values[0]))
        print('Dataframe File Location          : {}    '.format(Info_Dataframe[Info_Dataframe.Parameter=='Dataframe_File_Path'          ].Value.values[0]))
        print('---------------------------------------------')
    else:
        print('Empty Dataframe, File Will Not Be Saved')

if __name__ == "__version__":
    print
if __name__ == "__main__":
    InputArguments                     = GetArguments()
    ComponentList = list()
    StartTimer = datetime.now()
    TimeStamp = 'Start Time (hh:mm:ss.ms) {}'.format(StartTimer)
    print(TimeStamp)
    print('Begin housekeeping')
    SlideImageArray,\
    LabelMapImageArray  = Initialize()
    print('Image Deconvolution Begins')
    ComponentList       = ExecuteDeconvolution(SlideImageArray,LabelMapImageArray)
    print('Image Deconvolution Ends')
    Terminate(ComponentList)

    print('Wrap up time')
    TimeElapsed = datetime.now() - StartTimer
    TimeStamp   = 'Time elapsed (hh:mm:ss.ms) {}\n'.format(TimeElapsed)
    print(TimeStamp)