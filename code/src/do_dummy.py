import numpy as np
from skimage.io import imread
from PIL import Image

# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = None

# x = "data/for_normalization/image_maps/W1-1-2-E.1.01_11_LM_292324350.png"

# image = imread(x)

# ans = np.unique(image)

# print(ans)

# StainDictionary = {"idx":[1],
#                    "Hlabel":[2],
#                    "Elabel": [3]}
# import pandas as pd
# StainHistograms = pd.DataFrame(StainDictionary)

from Utilities import *
import numpy as np

img = Image.open('./data/for_normalization/Images/266290718.jpg')
img_rgb = img.convert('RGB')
img_array = np.array(img_rgb)
# print(img_array)
# stainvectoroutput = CalculateStainVector(img_array, 0.1, 240)
# print(stainvectoroutput)


NUMBER_OF_STAINS = 2
INFERRED_DIMENSION                = -1
NUMBER_OF_COLORS                  = 3

X = rgb2od(img).reshape((INFERRED_DIMENSION, NUMBER_OF_COLORS))

model = MiniBatchDictionaryLearning(n_components=NUMBER_OF_STAINS,
                                        alpha=0.1,
                                        max_iter=240,
                                        fit_algorithm='lars',
                                        positive_code=False,  # This option is not available, but post-processing could be applied
                                        transform_algorithm='lasso_lars',
                                        random_state=0)
print("Shape of X before fitting: ", X.shape)
model.fit(X)
print("Shape of X after fitting: ", X.shape)

# learned_dictionary = model.components_
# print(learned_dictionary)