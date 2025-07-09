#------------------------------------------------------------------------------------------------
# Name          : Utilities
#
# Description   : Set of functions to apply Vahadane's normalization algorithm
#                 https://github.com/wanghao14/Stain_Normalization
#
# ------------------------------------------------------------------------------------------------
# Library imports
# ------------------------------------------------------------------------------------------------
import numpy as np
import spams
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Lasso
import psutil
from tqdm import tqdm

WHITE_COLOR                       = 255
INFERRED_DIMENSION                = -1
NUMBER_OF_COLORS                  = 3

def od2rgb(OD):
    """
    Transforms from optical density to red-green-blue colorspace.
    :param OD: optical density matrix to be converted.
    :return: Same values but in RGB format.
    """
    return (WHITE_COLOR * np.exp(-1 * OD)).astype(np.uint8)

def rgb2od(Image):
    """
    Transforms an RGB image to the Optical Density (OD) colorspace.
    :param I: RGB image to be converted.
    :return: Same values but in RGB format.
    optical density is calculated as OD = -log(%t)
    OD = Log10(Io / I)
    """
    NewImage = np.where(Image == 0.0, 1, Image)
    Image_Adjusted = NewImage / WHITE_COLOR

    OpticalDensity = -np.log(Image_Adjusted)
    return OpticalDensity


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    print('Normalize Array')
    if np.all(A == 0.0):
        return A

    A = np.where(A == 0.0, 1/WHITE_COLOR, A)

    return A / np.linalg.norm(A, axis=1)[:, None]


def log_memory(stage=""):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"[Memory Log] {stage}: {mem:.2f} GB used")


#-----------------------------------------------------------------
# Name: Sort Out Stain Vectors
# Description: Output vectors definition is undefined. Need to find
#              H&E vector order by which is more blue.
#-----------------------------------------------------------------
def SortOutStainVectors(StainVectors):

    RED_COLOR            = 0
    GREEN_COLOR          = 1
    BLUE_COLOR           = 2
    FIRST_ITEM           = 0
    SECOND_ITEM          = 1
    print('Sort Out Stain Vectors')
    FirstStainRed        = StainVectors[FIRST_ITEM, RED_COLOR]
    FirstStainGreen      = StainVectors[FIRST_ITEM, GREEN_COLOR]
    FirstStainBlue       = StainVectors[FIRST_ITEM, BLUE_COLOR]

    SecondStainRed       = StainVectors[SECOND_ITEM, RED_COLOR]
    SecondStainGreen     = StainVectors[SECOND_ITEM, GREEN_COLOR]
    SecondStainBlue      = StainVectors[SECOND_ITEM, BLUE_COLOR]


    if FirstStainBlue >= SecondStainBlue:
        HematoxylinStainVector = [FirstStainRed,FirstStainGreen,FirstStainBlue]
        EosinStainVector       = [SecondStainRed,SecondStainGreen,SecondStainBlue]

    else:
        HematoxylinStainVector = [SecondStainRed,SecondStainGreen,SecondStainBlue]
        EosinStainVector       = [FirstStainRed,FirstStainGreen,FirstStainBlue]

    HandE_StainVectors = np.array([HematoxylinStainVector,EosinStainVector])

    return HandE_StainVectors


# def CalculateStainVector(Image,Stain_Vector_Lambda,Stain_Vector_Training_Time):
#     """
#     Get 2x3 stain matrix. First row H and second row E
#     :param I:
#     :param threshold:
#     :param ld:
#     :return:
#     Use smaller values of the lambda (0.01-0.1) for better reconstruction.
#     However, if the normalized image seems not fine, increase or decrease the value accordingly.
#     """
#     NUMBER_OF_STAINS = 2
#     X                = rgb2od(Image).reshape((INFERRED_DIMENSION, NUMBER_OF_COLORS))
#     StainVectors = spams.trainDL(
#         X        = X.T,
#         K        = NUMBER_OF_STAINS,
#         lambda1  = Stain_Vector_Lambda,
#         mode     = 2,
#         modeD    = 0,
#         posAlpha = True,
#         iter     = -Stain_Vector_Training_Time,
#         posD     = True,
#         verbose  = False
#     ).T
#     print('Normalize vector')
#     StainVectors      = normalize_rows(StainVectors)

#     print('need to find H&E order by looking at color ratios')
#     StainVectorOutput = SortOutStainVectors(StainVectors)

#     return StainVectorOutput


def CalculateStainVector(Image, Stain_Vector_Lambda, Stain_Vector_Training_Time, n_chunks=50, random_seed=0):
    """
    Get 2x3 stain matrix. First row H and second row E.
    :param Image: Input RGB image as a numpy array.
    :param Stain_Vector_Lambda: Sparsity controlling parameter.
    :param Stain_Vector_Training_Time: Not directly used due to scikit-learn's API, but you can adjust the 'n_iter' parameter of MiniBatchDictionaryLearning if needed.
    :return: A 2x3 matrix of stain vectors for Hematoxylin and Eosin.
    """
    NUMBER_OF_STAINS = 2
    if Image.shape[-1] != 3:
        # Drop the alpha channel
        Image = Image[..., :3]
    # Assuming rgb2od and other necessary functions are defined elsewhere
    OD = rgb2od(Image).reshape((INFERRED_DIMENSION, NUMBER_OF_COLORS))  # Reshape the optical density matrix
    n_pixels = OD.shape[0]
    chunk_size = (n_pixels + n_chunks - 1) // n_chunks

    # Initialize the MiniBatchDictionaryLearning model
    model = MiniBatchDictionaryLearning(n_components=NUMBER_OF_STAINS,
                                        alpha=Stain_Vector_Lambda,
                                        max_iter=Stain_Vector_Training_Time,
                                        fit_algorithm='lars',
                                        positive_code=False,  # This option is not available, but post-processing could be applied
                                        transform_algorithm='lasso_lars',
                                        random_state=random_seed)
    print('Starting to partial fit the model in chunks')
    for i in tqdm(range(n_chunks)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_pixels)
        if start >= end:
            break  # No more data to process
        OD_chunk = OD[start:end]
        model.partial_fit(OD_chunk)

    # # Fit the model and retrieve the components (stain vectors)
    # model.fit(OD)
    StainVectors = model.components_

    # Normalize each stain vector
    StainVectors = normalize_rows(StainVectors)

    # Post-process StainVectors to ensure positivity if necessary
    # For example: StainVectors[StainVectors < 0] = 0

    # Assuming SortOutStainVectors function is defined elsewhere and performs necessary post-processing
    StainVectorOutput = SortOutStainVectors(StainVectors)

    return StainVectorOutput


def CalculateDensityMap(Image, StainMatrix,lamda,n_chunks=50):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 six
    :return:
    """

    OD = rgb2od(Image).reshape(INFERRED_DIMENSION, NUMBER_OF_COLORS)
    if Image.shape[-1] != 3:
        # Drop the alpha channel
        Image = Image[..., :3]
    n_pixels = OD.shape[0]
    chunk_size = (n_pixels + n_chunks - 1) // n_chunks
    results = []
    print("Executing spams.lasso in chunks")
    for i in tqdm(range(n_chunks)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_pixels)
        if start >= end:
            break  # No more data to process
        OD_chunk = OD[start:end]
        DensityMapW = spams.lasso(X=OD_chunk.T, D=StainMatrix.T, lambda1=lamda, pos=True)
        results.append(DensityMapW.toarray().T)
    return np.vstack(results)

    # # Randomly sample pixels if too many
    # if n_pixels > max_samples:
    #     np.random.seed(random_seed)
    #     idx = np.random.choice(n_pixels, max_samples, replace=False)
    #     OD = OD[idx]

    # print('Executing spams.lasso')
    # DensityMapW           = spams.lasso(X       = OD.T,
    #                                     D       = StainMatrix.T,
    #                                     lambda1 = lamda,
    #                                     pos     = True)

    # return DensityMapW.toarray().T


# def CalculateDensityMap(Image, StainMatrix, lamda):
    # """
    # Get concentrations, a npix x 2 matrix
    # :param I:
    # :param stain_matrix: a 2x3 six
    # :return:
    # """

    # OD = rgb2od(Image).reshape(INFERRED_DIMENSION, NUMBER_OF_COLORS)

    # print('Executing Lasso')

    # model = Lasso(alpha=lamda, positive=True)
    # model.fit(OD.T, StainMatrix.T)
    # DensityMapW = model.coef_.T

    # return DensityMapW