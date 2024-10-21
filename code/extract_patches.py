import numpy as np
import h5py
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
from utils import save_file

Image.MAX_IMAGE_PIXELS = None

def extract_patches(images_folder, patch_size=256, hdf5_folder="data/for_normalization/patching", intensity_thresh=245):
    """
    Extracts patches of a given size from images in a folder and stores them, along with their coordinates,
    in HDF5 format. It also discards patches that contain more than a certain percentage of white pixels
    (based on an intensity threshold).

    Args:
        images_folder (str): The path to the folder containing the images.
        patch_size (int): The size of the patches to extract (default is 256x256 pixels).
        hdf5_folder (str): The folder to save the HDF5 files containing patches and coordinates.
        intensity_thresh (int): The pixel intensity threshold used to filter out mostly white patches.
                                Patches with more than 50% of pixels having intensity above this threshold
                                are discarded. If None, intensity thresholding is not applied.

    Steps:
        - Load each image.
        - Extract non-overlapping patches of size `patch_size`.
        - Filter out patches with more than 50% of white pixels.
        - Store the patches and their top-left coordinates in HDF5 format.
    """

    image_paths = glob.glob(os.path.join(images_folder, "*"))
    total = len(image_paths)
    for i, image_path in enumerate(tqdm(image_paths)):
        print(f"{i+1}/{total} : {image_path}")
        # Load the image
        image = Image.open(image_path)
        image = np.array(image)

        # Get image dimensions
        print(f"Image Shape: {image.shape}")
        height, width, _ = image.shape

        # Lists to store patches and coordinates
        patches = []
        coordinates = []

        # Loop through the image and extract patches along with their coordinates
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                # Extract the patch when both height and width exceed the image size
                if i + patch_size > height and j + patch_size > width:
                    patch = image[i:, j:, :]  # Extract remaining part of both dimensions
                    patch = np.pad(
                        patch,
                        ((0, max(0, patch_size - patch.shape[0])), (0, max(0, patch_size - patch.shape[1])), (0, 0)),
                        mode='constant',
                        constant_values=255
                    )

                # Extract the patch when only height exceeds the image size
                elif i + patch_size > height:
                    patch = image[i:, j:j + patch_size, :]  # Extract remaining part of height only
                    patch = np.pad(
                        patch,
                        ((0, max(0, patch_size - patch.shape[0])), (0, 0), (0, 0)),  # Pad only height
                        mode='constant',
                        constant_values=255
                    )

                # Extract the patch when only width exceeds the image size
                elif j + patch_size > width:
                    patch = image[i:i + patch_size, j:, :]  # Extract remaining part of width only
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, max(0, patch_size - patch.shape[1])), (0, 0)),  # Pad only width
                        mode='constant',
                        constant_values=255
                    )

                # Extract the patch when neither dimension exceeds the image size (no padding needed)
                else:
                    patch = image[i:i + patch_size, j:j + patch_size, :]  # Fully within bounds, no padding

                assert patch.shape == (256, 256, 3)

                # Discarding patches that have more than 50% of white pixels
                if intensity_thresh is not None:
                    count_white_pixels = np.where(
                        np.logical_and(
                            patch[:, :, 0] > intensity_thresh,
                            patch[:, :, 1] > intensity_thresh,
                            patch[:, :, 2] > intensity_thresh))[0]
                    # percentage of white pixels in the patch
                    percent_pixels = len(count_white_pixels) / (patch_size*patch_size)
                    if percent_pixels > 0.5:
                        continue

                patches.append(patch)
                coordinates.append((i, j))  # Stores the top-left corner (i, j) of the patch

        print(f"Total patches extracted: {len(patches)}")
        # Save patches and coordinates in HDF5 format
        image_name = image_path.split("/")[-1]
        image_name = image_name.split(".")[0]
        os.makedirs(hdf5_folder, exist_ok=True)

        with h5py.File(f"{os.path.join(hdf5_folder, image_name)}.h5", 'w') as h5f:
            h5f.create_dataset('imgs', data=np.array(patches))
            h5f.create_dataset('coords', data=np.array(coordinates), chunks=True, maxshape=(None, 2))

def stitch_patches(hdf5_file, original_size, patch_size=256):
    """
    Reconstructs an image from patches stored in an HDF5 file by stitching them back together
    based on their original coordinates.

    Args:
        hdf5_file (str): Path to the HDF5 file containing the patches and their coordinates.
        original_size (tuple): The original size of the image (height, width).
        patch_size (int): The size of the patches used to reconstruct the image (default is 256x256 pixels).

    Returns:
        Image: The reconstructed image as a PIL Image object.

    Steps:
        - Load the patches and their coordinates from the HDF5 file.
        - Create an empty image with the original dimensions.
        - Stitch the patches back into the empty image based on their coordinates.
    """
    # Load patches and coordinates from HDF5 file
    with h5py.File(hdf5_file, 'r') as h5f:
        patches = h5f['imgs'][:]
        coordinates = h5f['coords'][:]

    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)

    # Stitch patches back together using their coordinates
    for patch, (i, j) in zip(patches, coordinates):

        patch_height = min(patch_size, original_size[0] - i)
        patch_width = min(patch_size, original_size[1] - j)

        reconstructed_image[i:i+patch_height, j:j+patch_width, :] = patch[:patch_height, :patch_width, :]

    # Convert to PIL Image for saving or visualization
    return Image.fromarray(reconstructed_image)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract patches from WSI.')
    parser.add_argument('--images_folder', default="data/for_normalization/Images", type=str, required=True, help='Directory where the slide images are located.')
    parser.add_argument('--hdf5_folder', default="data/for_normalization/patches", type=str, required=True, help='Directory to store the extracted patches.')
    parser.add_argument('--patch_size', default=256, type=int, required=True, help='Patch size to extract')
    parser.add_argument('int_thresh', default=245, type=int, help='Pixel intensity threshold for discarging patches. If None, no patches will be discarded.')
    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = None

    images = os.listdir(args.images_folder)
    patch_size = 256

    # Step 1: Extract patches with coordinates
    extract_patches(args.images_folder, patch_size, args.hdf5_folder, args.int_thresh)

    # (Optional) Stitch patches back
    # reconstructed_image = stitch_patches(hdf5_file, original_size, patch_size)
    # save_file(reconstructed_image, reconstructed_image, ".jpg")