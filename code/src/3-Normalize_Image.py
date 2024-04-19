#------------------------------------------------------------------------------------------------
# Name          : Normalize Image
#
# Description   : This script normalizes image stain colors by applying
#                 a set of stain vectors and histogram
#
# ------------------------------------------------------------------------------------------------
# Library imports
# ------------------------------------------------------------------------------------------------
import dask
import spams
import argparse
import itertools
import Utilities
import skimage.io
import numpy               as np
import dask.array          as da
from PIL                   import Image
from datetime              import datetime
from pathlib               import Path
from skimage.color         import rgb2lab
from skimage.exposure      import match_histograms
from dask_image.imread     import imread
from sklearn.preprocessing import minmax_scale
#
__version__ = "0.0.1"
#------------------------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS                      = None
HEMATOXYLIN_STAIN                           = 0
EOSIN_STAIN                                 = 1
COLOR                                       = 1
PIXEL_QUANTITY                              = 0
#------------------------------------------------------------------------------------------------
# Function Name: Get Arguments
# Description: Define input arguments
# Input: image name, histogram, stain vectors and output directory
# Output: Argument list
#------------------------------------------------------------------------------------------------
def GetArguments():
    DESCRITPTION_MESSAGE = \
    'This scripts normalizes provided image using previously found histogram and stain vectors.\n' +\
    'The Normalizing histogram is a tuple with two numpy arrays. The first array holds color   \n' +\
    'info (float) and the second the pixel count (int). There two histograms per npy file,     \n' +\
    'the first is Hematoxylin: Color, pixel count, and the 2nd Eosin: Color, pixel count       \n' +\
    'The stain vectors is an array of three RGB values, the first Hematoxylin and the 2nd Eosin\n' +\
    '                                                                                          \n' +\
    'usage:                                                                                    \n' +\
    'Normalize_Image.py                                                                        \n' +\
    '      --Image_To_Normalize        266290664.jpg                                           \n' +\
    '      --Normalizing_Histogram     100ImageCohortHistograms.npy                            \n' +\
    '      --Normalizing_Stain_Vectors 100ImageCohortStainVectors.npy                          \n' +\
    '      --Output_Directory          Normalized_Images_Directory                             \n'

    parser = argparse.ArgumentParser(description=DESCRITPTION_MESSAGE)
    parser.add_argument('-i', '--Image_To_Normalize',       required=True,                  help='Image to Normalize')
    parser.add_argument('-n', '--Normalizing_Histogram',    required=True,                  help='Normalizing Histogram numpy')
    parser.add_argument('-s', '--Normalizing_Stain_Vectors',required=True,                  help='Normalizing Stain Vector numpy')
    parser.add_argument('-o', '--Output_Directory',         required=True,                  help='Output Directory')
    parser.add_argument('-t', '--Stain_Vector_Training',    required=False, default=240,    help='Stain Vector Training time in seconds')
    parser.add_argument('-l', '--Stain_Vector_Lambda',      required=False, default=0.1,    help='Stain Vector Lambda')
    parser.add_argument('-d', '--Density_Map_Lambda',       required=False, default=0.01,   help='Density Map Lambda')

    parser.add_argument('-v', '--version', action='version', version= "%(prog)s (ver: "+__version__+")")

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------------------------
# Function Name: Initialize
# Description: Sets up input, directories, and images for process
# Input: None
# Output: Output File path
#------------------------------------------------------------------------------------------------
def Initialize():
    global InputArguments

    OUTPUT_IMAGE_POSTFIX      = '_Normalized'
    # FILE_EXTENSION            = '.png'
    FILE_EXTENSION            = '.tif'
    InputImageName            = str()
    InputImageNameNoExtension = str()
    NewFileName               = str()
    OutputFilePath            = str()

    print('----------------------------------------------------')
    print('Initialization Begins')

    InputImageName            = Path(InputArguments.Image_To_Normalize).name
    InputImageNameNoExtension = Path(InputImageName).stem
    NewFileName               = InputImageNameNoExtension + OUTPUT_IMAGE_POSTFIX + FILE_EXTENSION

    OutputFilePath            = str(Path(InputArguments.Output_Directory) / NewFileName)

    print('----------------------------------------------------')

    return OutputFilePath
#------------------------------------------------------------------------------------------------
# Function Name: Creates a directory
# Description: Created a directory
# Input: path
# Output: none
#------------------------------------------------------------------------------------------------
def CreateDirectory(OutputPath):
    try:
        print('Creating directory:\n{}'.format(OutputPath))
        Path(OutputPath).mkdir(parents=True, exist_ok=True)
    except:
        print('Could not created directory:\n{}'.format(OutputPath))
        raise IOError()

#------------------------------------------------------------------------------------------------
# Function Name: Load Input Files
# Description: Sets up input, directories, and images for process
# Input: None
# Output: Image name, histogram, and stain vectors
#------------------------------------------------------------------------------------------------
def LoadInputFiles():
    global InputArguments
    DASK_DIMENSION   = 0
    ImageToNormalize = da.from_array(np.array([]))
    FileName         = str()
    print('Load image, histograms, and stain vectors')
    try:
        FileName                = str(Path(InputArguments.Image_To_Normalize).name)
        print('Load RGB Image to Normalize (dask): {}'.format(FileName))
        ImageToNormalize        = imread(InputArguments.Image_To_Normalize)
        print('Pop additional dimension inherent to dask')
        ImageToNormalize        = ImageToNormalize[DASK_DIMENSION] # Get rid of inherent dask dimension
    except:
        raise IOError('Unable to load:\n{}'.format(FileName))
    try:
        FileName                = str(Path(InputArguments.Normalizing_Stain_Vectors).name)
        print('Load Stain Vectors: {}'.format(FileName))
        NormalizingStainVectors = np.load(InputArguments.Normalizing_Stain_Vectors)
    except:
        raise IOError('Unable to load:  {}'.format(FileName))
    try:
        FileName                = str(Path(InputArguments.Normalizing_Histogram).name)
        NormalizingHistogram    = np.load(InputArguments.Normalizing_Histogram,    allow_pickle=True)
    except:
        raise IOError('Unable to load:\n{}'.format(FileName))

    # Update user ----------------------------
    print(' Loading Input Files')
    print(' Image Name        : {}'.format(Path(InputArguments.Image_To_Normalize).name))
    print(' Image Shape       : {}'.format(ImageToNormalize.shape))
    print(' Histogram Name    : {}'.format(Path(InputArguments.Normalizing_Histogram).name))
    print('     Data Length   : {}'.format(len(NormalizingHistogram[HEMATOXYLIN_STAIN][PIXEL_QUANTITY])))
    print('     Bin  Length   : {}'.format(len(NormalizingHistogram[HEMATOXYLIN_STAIN][COLOR])))
    print(' Stain Vectors Name: {}'.format(Path(InputArguments.Normalizing_Stain_Vectors).name))
    print('     H Vectors     : {}'.format(NormalizingStainVectors[HEMATOXYLIN_STAIN]))
    print('     E Vectors     : {}'.format(NormalizingStainVectors[EOSIN_STAIN]))

    return ImageToNormalize,NormalizingHistogram,NormalizingStainVectors

#------------------------------------------------------------------------------------------------
# Function Name: Terminate
# Description: Save normalized image and update user
# Input: output file name, normalized image, input image, stain vectors
# Output: None
#------------------------------------------------------------------------------------------------
def Terminate(OutputFilePath,NewImage, OriginalImage,NormalizingStainVectors):
    global InputArguments

    if len(NewImage):

        print('Save image as a png file')
        skimage.io.imsave(OutputFilePath, NewImage)

        print('----------------------------------------------------')
        print('Stain Vectors:')
        print('  Hematoxylin       : {}'.format(NormalizingStainVectors[HEMATOXYLIN_STAIN]))
        print('  Eosin             : {}'.format(NormalizingStainVectors[EOSIN_STAIN]))
        print('Normalized Image Summary')
        print('  File Image Name   : {}'.format(Path(OutputFilePath).name))
        print('  Image Shape       : {}'.format(NewImage.shape))
        print('  Output Directory  : {}'.format(Path(OutputFilePath).parent))
        print('  Histogram Name    : {}'.format(Path(InputArguments.Normalizing_Histogram).name))
        print('  Stain Vectors Name: {}'.format(Path(InputArguments.Normalizing_Stain_Vectors).name))
        print('Original Image Summary')
        print('  File Image Name   : {}'.format(str(Path(InputArguments.Image_To_Normalize).name)))
        print('  Image Shape       : {}'.format(OriginalImage.shape))
        print('----------------------------------------------------')
    else:
        raise IOError('Error: Empty Normalized Image')

#-----------------------------------------------------------------
# Name: Sort Out Stain Vectors
# Description: Output vectors definition is undefined. Need to find
#              H&E vector order by which is more blue.
# Input: Stain vectors
# Output: sorted stain vectors
#-----------------------------------------------------------------
def SortOutStainVectors(StainVectors):
    RED_COLOR        = 0
    GREEN_COLOR      = 1
    BLUE_COLOR       = 2
    FIRST_ITEM       = 0
    SECOND_ITEM      = 1
    print('Sort Out Stain Vectors')
    FirstStainRed    = StainVectors[FIRST_ITEM, RED_COLOR]
    FirstStainGreen  = StainVectors[FIRST_ITEM, GREEN_COLOR]
    FirstStainBlue   = StainVectors[FIRST_ITEM, BLUE_COLOR]

    SecondStainRed   = StainVectors[SECOND_ITEM, RED_COLOR]
    SecondStainGreen = StainVectors[SECOND_ITEM, GREEN_COLOR]
    SecondStainBlue  = StainVectors[SECOND_ITEM, BLUE_COLOR]

    if FirstStainBlue >= SecondStainBlue:
        HematoxylinStainVector = [FirstStainRed,FirstStainGreen,FirstStainBlue]
        EosinStainVector       = [SecondStainRed,SecondStainGreen,SecondStainBlue]

    else:
        HematoxylinStainVector = [SecondStainRed,SecondStainGreen,SecondStainBlue]
        EosinStainVector       = [FirstStainRed,FirstStainGreen,FirstStainBlue]

    HandE_StainVectors = np.array([HematoxylinStainVector,EosinStainVector])

    return HandE_StainVectors

#------------------------------------------------------------------------------------------------
# Function Name: Use Histogram Matching
# Description: Match histogram
# Input: image optical density, color array, and pixel count
# Output: Histogram
#------------------------------------------------------------------------------------------------
def UseHistogramMatching(ImageToTransformDensity, PixelColorArray,PixelCountArray):
    #-----------------------------------------------------------------
    print('Unravel Histogram')
    UnraveledData               = ImprovedUnravelArray(PixelColorArray,PixelCountArray)
    print('Get rid of possible stray zeros')
    UnraveledData               = UnraveledData[np.nonzero(UnraveledData)]
    #-----------------------------------------------------------------
    Normalize_Reference_Density = minmax_scale(UnraveledData,feature_range=(ImageToTransformDensity.min(), ImageToTransformDensity.max()))
    print('Match histograms')
    MatchedHistogramDensity = match_histograms(ImageToTransformDensity[...,None],\
                                               Normalize_Reference_Density[...,None],\
                                               channel_axis=-1)
    return MatchedHistogramDensity[:,0]
#------------------------------------------------------------------------------------------------
# Function Name: Improved Unravel Array
# Description: Unravel histogram
# Input: color array, and pixel count
# Output: Array
#------------------------------------------------------------------------------------------------
def ImprovedUnravelArray(PixelColorArray,PixelCountArray):

    MyList = list()
    for PixelColor,PixelCount in zip(PixelColorArray,PixelCountArray):
        MyList += int(PixelCount)*[PixelColor]

    OutputArray = np.array(list(itertools.chain.from_iterable([int(PixelCount)*[PixelColor] for PixelColor,PixelCount in zip(PixelColorArray,PixelCountArray)])))
    del PixelColorArray,PixelCountArray
    return OutputArray

#------------------------------------------------------------------------------------------------
# Function Name: Normalize Image
# Description: Normalize image
# Input: image, histogram, and stain vectors
# Output: normalized image
#------------------------------------------------------------------------------------------------
def NormalizeImage(InputImage,NormalizingHistogram,NormalizingStainVectors):

    WHITE_COLOR         = 255
    INFERRED_DIMENSION  = -1
    NUMBER_OF_COLORS    = 3
    LUMINANCE_THRESHOLD = 90

    print('Find Image Lab Space')
    LabColorSpaceImage = rgb2lab(InputImage)
    print('Extract Luminance from Lab Space')
    Luminance          = LabColorSpaceImage[...,0].astype(np.uint8)
    print('Mask Image based on {} Luminance'.format(LUMINANCE_THRESHOLD))
    LuminanceMask      = Luminance<LUMINANCE_THRESHOLD
    print('NotWhite Mask Housekeeping')
    #------------------------------------------------------------------------
    print('Flatten Image and extract Useful Pixels')
    Width,Height,RGB     = InputImage.shape
    FlattenImage         = InputImage.reshape([Width * Height,RGB])
    LuminanceMaskFlatten = LuminanceMask.flatten()
    dask.config.set({"array.slicing.split_large_chunks": True})
    ExtractedPixels      = FlattenImage[LuminanceMaskFlatten]
    if ExtractedPixels.size:
        print('-----------------------------------')
        print('Convert Image from RGB to OD')
        ExtractedPixels      = da.where(ExtractedPixels == 0, 1, ExtractedPixels)
        print('Computing -1 * Log')
        ODImage              = -1 * da.log(ExtractedPixels / WHITE_COLOR).compute()

        print('-----------------------------------')
        print('Find Stain Vectors using Image Optical Density')
        SlideStainVectors    = FindStainVectors(ODImage.T)
        RgbStainVectors      = Utilities.od2rgb(SlideStainVectors)
        print('------------------------------------')
        print('Stain Vectors')
        print(f'\tHematoxylin : {RgbStainVectors[0]}')
        print(f'\tEosin       : {RgbStainVectors[1]}')
        print('-----------------------------------')
        print('Calculate Density Map using Not White Image Optical Density')
        StainDensityMap      = FindDensityMap(ODImage.T,SlideStainVectors)

        print('Find Non-Zero Elements')
        ImageSelection       = np.argwhere(LuminanceMask)
        print('ImageSelection Size: {}'.format(ImageSelection.shape))
        ImageMaskIndeces     = ImageSelection[:, 0]

        print('-----------------------------------')
        StainLocation          = HEMATOXYLIN_STAIN
        StainName              = 'Hematoxylin'
        print('{}: Build transformed histogram '.format(StainName))
        print('{}: Apply Normalizing Histogram '.format(StainName))
        DensityMap             = StainDensityMap[:, StainLocation]
        print('{}: Set Negative Values to 0    '.format(StainName))
        StainMask              = DensityMap > 0
        print('{}: Unravel Mask Indeces        '.format(StainName))
        StainMaskIndexes       = np.argwhere(StainMask).flatten()
        print('{}: Apply Mask                  '.format(StainName))
        TrimmedDensityMap      = DensityMap[StainMask]
        print('{}: Match Histogram             '.format(StainName))
        Bins = NormalizingHistogram[StainLocation][COLOR]
        print('{}: Histogram Bins: {}          '.format(StainName, Bins.shape))
        Data                   = NormalizingHistogram[StainLocation][PIXEL_QUANTITY]
        print('{}: Histogram Data: {}          '.format(StainName, Data.shape))
        TransformedStain       = UseHistogramMatching(TrimmedDensityMap,Bins,Data)
        print('{}: Apply new stain densities  '.format(StainName))
        DensityMap[StainMaskIndexes] = TransformedStain
        print('{}: Check for negative numbers '.format(StainName))
        TransformedHematoxylin = np.where(DensityMap < 0,0,DensityMap)

        print('-----------------------------------')
        StainLocation         = EOSIN_STAIN
        StainName             = 'Eosin'
        print('{}: Build transformed histogram '.format(StainName))
        print('{}: Apply Normalizing Histogram '.format(StainName))
        DensityMap         = StainDensityMap[:, StainLocation]
        print('{}: Set Negative Values to 0    '.format(StainName))
        StainMask          = DensityMap > 0
        print('{}: Unravel Mask Indeces        '.format(StainName))
        StainMaskIndexes   = np.argwhere(StainMask).flatten()
        print('{}: Apply Mask                  '.format(StainName))
        TrimmedDensityMap  = DensityMap[StainMask]
        print('{}: Match Histogram             '.format(StainName))
        Bins = NormalizingHistogram[StainLocation][COLOR]
        print('{}: Histogram Bins: {}          '.format(StainName, Bins.shape))
        Data                   = NormalizingHistogram[StainLocation][PIXEL_QUANTITY]
        print('{}: Histogram Data: {}          '.format(StainName, Data.shape))
        TransformedStain   = UseHistogramMatching(TrimmedDensityMap,Bins,Data)
        print('{}: Apply new stain densities  '.format(StainName))
        DensityMap[StainMaskIndexes] = TransformedStain
        print('{}: Check for negative numbers '.format(StainName))
        TransformedEosin      = np.where(DensityMap < 0,0,DensityMap)

        print('-----------------------------------')
        print('Build new image using transformed histograms')
        print('Hematoxylin Array Shape: {}'.format(TransformedEosin.shape))
        print('Eosin Array Shape      : {}'.format(TransformedHematoxylin.shape))
        print('-----------------------------------')

        print('Convert Image from RGB to OD')
        print('Replacing 0 for 1/White Color')
        OutputImage           = np.where(InputImage == 0.0, 1/WHITE_COLOR, InputImage)
        print('Normalizing Image to White Color')
        OutputImage           = np.divide(OutputImage,WHITE_COLOR)
        print('Computing the -1 * Log and Converting Dask to Numpy')
        ODImage               = np.negative(np.log(OutputImage))
        print('Reshaping Array')
        ODImage               = ODImage.reshape(INFERRED_DIMENSION, NUMBER_OF_COLORS)
        print('Calculate Density Map')
        StainDensityMapDask  = FindDensityMap(ODImage.compute().T,SlideStainVectors)

        print('Stacking Stain Arrays into Density Map')
        StainDensityMapDask[ImageMaskIndeces] = np.column_stack((TransformedEosin,TransformedHematoxylin))
        print('-----------------------------------')
        print('Apply normalizing stain vectors to image')
        print('Replacing Stain Vector 0 for 1/{}'.format(WHITE_COLOR))
        ODStainVectors      = np.where(NormalizingStainVectors == 0.0, 1/WHITE_COLOR, NormalizingStainVectors)
        print('Normalizing Stain Vector to White Color')
        ODStainVectors      = np.divide(ODStainVectors,WHITE_COLOR)
        print('Computing Stain Vector Log')
        ODStainVectors      = np.log(ODStainVectors)
        print('Computing Stain Vector -1 * Log and Converting Dask to Numpy')
        ODStainVectors      = np.negative(ODStainVectors)
        print('Incorpotating Stain Vectors into Density Map')
        ColorMatrix         = np.dot(StainDensityMapDask, ODStainVectors)

        print('Converting Output Array RGB and Reshape')
        OutputImage         = WHITE_COLOR * np.exp(-1 * ColorMatrix.reshape(InputImage.shape))
        print('Casting Array to UINT8')
        OutputImage         = OutputImage.astype(np.uint8)
    else:
        print('No Pixels Avaialble to Normalize. Moving on ...')
        OutputImage=InputImage

    return OutputImage
#------------------------------------------------------------------------------------------------
# Function Name: Find Stain Vectors
# Description:
# Input: None
# Output: None
#------------------------------------------------------------------------------------------------
def FindStainVectors(Data):
    NUMBER_OF_STAINS=2
    StainVectors = spams.trainDL(
        X        = Data,
        K        = NUMBER_OF_STAINS,
        lambda1  = float(InputArguments.Stain_Vector_Lambda),
        mode     = 2,
        modeD    = 0,
        posAlpha = True,
        iter     = -int(InputArguments.Stain_Vector_Training),
        posD     = True,
        verbose  = True
    ).T

    print('Normalize vector')
    StainVectors      = normalize_rows(StainVectors)

    print('need to find H&E order by looking at color ratios')
    StainVectorOutput = SortOutStainVectors(StainVectors)

    return StainVectorOutput

def normalize_rows(InputArray):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    WHITE_COLOR         = 255

    NewImage = da.where(InputArray == 0, 1/WHITE_COLOR, InputArray)

    return NewImage / da.linalg.norm(NewImage, axis=1)[:, None]
#------------------------------------------------------------------------------------------------
# Function Name: Find Density Map
# Description:
# Input: None
# Output: None
#------------------------------------------------------------------------------------------------
def FindDensityMap(OpticalDensityImage,SlideStainVectors):
    StainDensityMap = spams.lasso(X       = OpticalDensityImage,
                                  D       = SlideStainVectors.T,
                                  lambda1 = float(InputArguments.Density_Map_Lambda),
                                  pos     = True)
    return StainDensityMap.toarray().T
#------------------------------------------------------------------------------------------------
if __name__ == "__version__":
    print
if __name__ == "__main__":

    InputArguments          = GetArguments()

    StartTimer = datetime.now()
    TimeStamp  = 'Start Time (hh:mm:ss.ms) {}'.format(StartTimer)
    print(TimeStamp)
    #------------------------------------------------------------------------------------------------
    NormalizedImage         = da.from_array([], chunks='200MiB')

    OutputFilePath          = Initialize()

    ImageToNormalize,\
    NormalizingHistogram,\
    NormalizingStainVectors = LoadInputFiles()

    NormalizedImage         = NormalizeImage(ImageToNormalize,NormalizingHistogram,NormalizingStainVectors)

    Terminate(OutputFilePath,NormalizedImage,ImageToNormalize,NormalizingStainVectors)
    #------------------------------------------------------------------------------------------------
    TimeElapsed = datetime.now() - StartTimer
    TimeStamp   = 'Time elapsed (hh:mm:ss.ms) {}\n'.format(TimeElapsed)
    print(TimeStamp)