import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob
from PIL import Image
from skimage.color import rgb2lab
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None


def plot_comparisons(org, norm_with_org_mask, norm_with_our_mask, output_folder, resize_shape=(400, 300)):
    """
    Visualizes images from three lists (original, normalized with original mask,
    normalized with our mask) in both RGB and CIELAB color spaces, with resizing.

    Args:
        org (list of str): List of file paths for original images.
        norm_with_org_mask (list of str): List of file paths for images normalized with original mask.
        norm_with_our_mask (list of str): List of file paths for images normalized with our mask.
        output_folder (str): Path to the folder where the output plots will be saved.
        resize_shape (tuple): Target size to resize images for plotting (height, width).

    Notes:
        - Each row in the subplot shows one image set in the order: original, normalized with original mask,
          and normalized with our mask.
        - Two subplots are created: one for RGB visualization and one for CIELAB visualization.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Number of images
    num_images = len(org)

    # Plot in RGB color space
    # fig_rgb, axes_rgb = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    # fig_rgb.suptitle('Comparison of Images in RGB Color Space', fontsize=20)

    # Plot in CIELAB color space
    fig_lab, axes_lab = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    fig_lab.suptitle('Comparing Images in CIELAB Color Space', fontsize=20)

    for i in range(num_images):
        print(f"Image: {i+1}")
        # Read and resize images
        org_img = resize(imread(org[i]), resize_shape, anti_aliasing=True)
        norm_org_mask_img = resize(imread(norm_with_org_mask[i]), resize_shape, anti_aliasing=True)
        norm_our_mask_img = resize(imread(norm_with_our_mask[i]), resize_shape, anti_aliasing=True)
        print(f"Images resized to {org_img.shape}")

        # # RGB Plotting
        # axes_rgb[i, 0].imshow(org_img)
        # axes_rgb[i, 0].set_title("Original (RGB)")
        # axes_rgb[i, 1].imshow(norm_org_mask_img)
        # axes_rgb[i, 1].set_title("Normalized (Original Mask, RGB)")
        # axes_rgb[i, 2].imshow(norm_our_mask_img)
        # axes_rgb[i, 2].set_title("Normalized (Our Mask, RGB)")

        # Convert images to CIELAB
        org_img_lab = rgb2lab(org_img)
        norm_org_mask_img_lab = rgb2lab(norm_org_mask_img)
        norm_our_mask_img_lab = rgb2lab(norm_our_mask_img)
        print(f"Images converted to CEILAB space")

        # Normalize CIELAB channels for display
        org_img_lab_display = (org_img_lab - org_img_lab.min()) / (org_img_lab.max() - org_img_lab.min())
        norm_org_mask_img_lab_display = (norm_org_mask_img_lab - norm_org_mask_img_lab.min()) / (norm_org_mask_img_lab.max() - norm_org_mask_img_lab.min())
        norm_our_mask_img_lab_display = (norm_our_mask_img_lab - norm_our_mask_img_lab.min()) / (norm_our_mask_img_lab.max() - norm_our_mask_img_lab.min())
        print(f"Images normalized for CEILAB")

        # CIELAB Plotting
        axes_lab[i, 0].imshow(org_img_lab_display)
        axes_lab[i, 0].set_title("Original Image")
        axes_lab[i, 1].imshow(norm_org_mask_img_lab_display)
        axes_lab[i, 1].set_title("Normalized Image with Original Mask")
        axes_lab[i, 2].imshow(norm_our_mask_img_lab_display)
        axes_lab[i, 2].set_title("Normalized Image with Our Mask")

    # # Adjust layout and save RGB plot
    # for ax in axes_rgb.ravel():
    #     ax.axis('off')
    # fig_rgb.tight_layout(rect=[0, 0, 1, 0.96])
    # rgb_output_path = os.path.join(output_folder, 'comparison_rgb.png')
    # fig_rgb.savefig(rgb_output_path, dpi=300)
    # plt.close(fig_rgb)

    # Adjust layout and save CIELAB plot
    for ax in axes_lab.ravel():
        ax.axis('off')
    fig_lab.tight_layout(rect=[0, 0, 1, 0.96])
    lab_output_path = os.path.join(output_folder, 'comparison_lab.png')
    fig_lab.savefig(lab_output_path, dpi=300)
    plt.close(fig_lab)

    # print(f"RGB comparison plot saved at: {rgb_output_path}")
    print(f"CIELAB comparison plot saved at: {lab_output_path}")


def compare_pi_std_by_roi(image_folder,
                          mask_folder,
                          output_folder,
                          tag,
                          num_rois):
    """
    Computes and visualizes the standard deviation of pixel intensities across multiple regions of interest (ROIs)
    in a set of images, based on corresponding mask regions, and saves a boxplot summarizing these statistics.

    Steps:
        - Gather a list of images and masks from `image_folder` and `mask_folder`, matching them based on a shared
          base name (e.g., "image_001_Normalized.png" matches with "image_001.png").
        - For each matched image-mask pair:
            - Extract pixel intensities for each ROI in the image, based on the mask labels.
            - Compute the standard deviation of pixel intensities within each ROI.
            - Collect these statistics for all images.
        - Generate a boxplot showing the distribution of standard deviations by ROI.
        - Save the plot in the specified `output_folder`.

    Args:
        image_folder (str): The path to the folder containing the images.
                            Image filenames should partially match those of the corresponding masks.
        mask_folder (str): The path to the folder containing the mask files.
                           Each mask should label different ROIs as integer values (e.g., 0, 1, 2).
        output_folder (str): The path to the folder where the generated boxplot will be saved.
                             The folder will be created if it does not exist.
        tag (str): A descriptive label to add to the plot title and output filename, helping
                   identify different sets of images.
        num_rois (int): The total number of distinct ROIs expected in each mask (e.g., 5 for labels 0-4).
                        Assumes that all masks use consistent labeling.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    std_devs_by_roi = [[] for _ in range(num_rois)]

    mask_files = glob(os.path.join(mask_folder, "*"))

    for mask_file in mask_files:
        name = mask_file.split("/")[-1]
        base_name = name.split(".")[0]
        image_name = base_name + "_Normalized.png"  # Get base name of image
        image_file = os.path.join(image_folder, image_name)

        print(f"{image_file} - {mask_file}")
        image = imread(image_file, as_gray=True)  # Read as grayscale
        mask = imread(mask_file)  # Mask should be integer labeled regions

        for roi in range(num_rois+1):
            roi_pixels = image[mask == roi]  # Extract pixels for this ROI
            if len(roi_pixels) > 0:
                std_dev = np.std(roi_pixels)
                std_devs_by_roi[roi].append(std_dev)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.boxplot(std_devs_by_roi,
                positions=range(num_rois),
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"))

    # Add individual points
    for roi, std_devs in enumerate(std_devs_by_roi):
        plt.plot([roi] * len(std_devs), std_devs, 'ro', markersize=5, alpha=0.6)

    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title(f"Standard Deviation of Pixel Intensities by ROI - {tag}")
    plt.xticks(range(num_rois), [f"Region {i}" for i in range(num_rois)])

    # Save the figure
    output_path = os.path.join(output_folder, f'std_dev_plot_{tag}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {output_path}")


if __name__ == "__main__":

    # image_folder2 = './data/for_normalization/Images'
    image_folder = "./results/clustering/Normalized_Images"
    image_folder2 = "./results/Normalized_Images"
    mask_folder = './data/for_normalization/KM_Masks/'
    mask_folder2 = './data/for_normalization/Image_Maps/'
    num_roi = 8
    num_roi2 = 11
    tag = 'Normalized_Image_With_Our_Mask'
    tag2 = 'Normalized_Image_With_Original_Mask'
    output_folder = './results/plots'

    # compare_pi_std_by_roi(image_folder, mask_folder, output_folder, tag, num_roi)
    # compare_pi_std_by_roi(image_folder2, mask_folder2, output_folder, tag2, num_roi2)

    org_img = ["./data/for_normalization/Images/266291365.jpg",
               "./data/for_normalization/Images/294219481.jpg",
               "./data/for_normalization/Images/310443139.jpg"]

    norm_with_org_mask = ["./results/Normalized_Images/266291365_Normalized.png",
                          "./results//Normalized_Images/294219481_Normalized.png",
                          "./results/Normalized_Images/310443139_Normalized.png"]

    norm_with_our_mask = ["./results/clustering/Normalized_Images/266291365_Normalized.png",
                          "./results/clustering/Normalized_Images/294219481_Normalized.png",
                          "./results/clustering/Normalized_Images/310443139_Normalized.png"]

    plot_comparisons(org_img, norm_with_org_mask, norm_with_our_mask, output_folder)