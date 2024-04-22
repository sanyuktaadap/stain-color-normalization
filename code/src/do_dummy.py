import numpy as np
from skimage.io import imread
from PIL import Image

# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = None

# -----------See unique pixel values------------
x = "data/for_normalization/image_maps/W1-1-2-E.1.01_11_LM_292324350.png"
# Read image
image = imread(x)
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