import numpy as np
from skimage.io import imread
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = None

-----------See unique pixel values------------
x = "data/for_normalization/image_maps/W9-1-1-H.2.02_41_LM_309714697.png"
# Read image
image = imread(x)
img2 = image.shape
# Extract unique values
ans = np.unique(image)
# print unique values
print(ans)

# ------------Convert svs image to dask RGB array-------------
import openslide
import dask.array as da
import numpy as np
# Open the SVS image using openslide
slide = openslide.OpenSlide('data/temp/TCGA-12-1089-01A-01-TS1.7c4d6265-161f-4c71-ae8e-4ccab1e86a9e.svs')
# Read the image as a NumPy array
image_array = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]))
# Convert NumPy array to Dask array
dask_array = da.from_array(image_array)
# Extract only RGB channels from RGBA
dask_array = dask_array[:, :, :3]

# --------------Compare 2 images-----------------
masks_path = "./data/for_comparison/Image_Maps"
masks_list = os.listdir(masks_path)

# load map
mask_path = "data/for_comparison/Image_Maps/W9-1-1-E.2.02_11_LM_292343048.png"
mask = imread(mask_path)

lis1 = np.array([[0,2,2,1],[3,2,1,1],[1,0,0,0]])
indices = np.where(lis1 == 1) # 2 arrays-- one for x coordinates and one for y coordinates
loc = list(zip(indices[0], indices[1])) # list of tuples of with location of (x,y) coordinates

all_imgs_dir = "data/"

# ----------------Get selected images--------------

import os
import shutil
import numpy as np
from skimage.io import imread
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = None

norm_images_path = "data/for_comparison/Normalized_Images"
norm_images_list = os.listdir(norm_images_path)
images_path = "data/Images"
images_list = os.listdir(images_path)
comp_images_path = "data/for_comparison/Images/"

for norm_image in norm_images_list:
    number = norm_image.split("_")[0]
    name = number + ".jpg"
    if name in images_list:
        shutil.copy(os.path.join(images_path, name), comp_images_path)

import pandas as pd

csv_path = "data/Csv_Files/VerifiedIvyGap858ImageCohort.csv"
maps_path = "data/Image_Maps/"
comp_maps_path = "data/for_comparison/Image_Maps"
norm_images_path = "data/for_comparison/Normalized_Images"
norm_images_list = os.listdir(norm_images_path)

data = pd.read_csv(csv_path, header=None)
data = data.iloc[:,0]

for row in data:
    map_name = row.split("/")[-1]
    img = map_name.split("_")[-1].split(".")[0]
    name = img + "_Normalized.png"
    if name in norm_images_list:
        shutil.copy(os.path.join(maps_path, map_name), comp_maps_path)