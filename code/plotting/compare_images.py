import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob
from PIL import Image
import seaborn as sns
from skimage.color import rgb2hed
import pandas as pd
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def plot_comparisons(org, norm_with_org_mask, norm_with_our_mask, output_folder):
    """
    Visualizes images from three lists (original, normalized with original mask,
    normalized with our mask) in RGB color spaces.

    Args:
        org (list of str): List of file paths for original images.
        norm_with_org_mask (list of str): List of file paths for images normalized with original mask.
        norm_with_our_mask (list of str): List of file paths for images normalized with our mask.
        output_folder (str): Path to the folder where the output plots will be saved.

    Notes:
        - Each row in the subplot shows one image set in the order: original, normalized with original mask,
          and normalized with our mask.
        - Subplots are created for RGB visualization.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Number of images
    num_images = len(org)

    # Plot in RGB color space
    fig_rgb, axes_rgb = plt.subplots(num_images,
                                     3,
                                     figsize=(15, 30))

    fig_rgb.suptitle('Comparison of Images in RGB Color Space', fontsize=24)

    # Set column titles
    column_titles = ["Original", "Normalized With Original Mask", "Normalized with Our Mask"]
    for col, title in enumerate(column_titles):
        axes_rgb[0, col].set_title(title, fontsize=24, pad=20)

    for i in tqdm(range(num_images)):
        print(f"Image: {i+1}")

        org_img = imread(org[i])
        norm_org_mask_img = imread(norm_with_org_mask[i])
        norm_our_mask_img = imread(norm_with_our_mask[i])

        # RGB Plotting
        axes_rgb[i, 0].imshow(org_img)
        axes_rgb[i, 1].imshow(norm_org_mask_img)
        axes_rgb[i, 2].imshow(norm_our_mask_img)

    # Adjust layout and save RGB plot
    for ax in axes_rgb.ravel():
        ax.axis('off')
    fig_rgb.tight_layout(rect=[0, 0, 1, 0.96])
    rgb_output_path = os.path.join(output_folder, 'comparison_rgb.png')
    fig_rgb.savefig(rgb_output_path, dpi=200)
    plt.close(fig_rgb)

    print(f"RGB comparison plot saved at: {rgb_output_path}")

def compare_pi_std_by_roi(image_folders=["data/for_normalization/Images",
                                         "results/Normalized_Images",
                                         "results/clustering/Normalized_Images"],
                          mask_folder="data/for_normalization/Image_Maps",
                          num_rois=11,
                          output_folder="results/plots"):
    """
    Computes and visualizes the standard deviation of pixel intensities within different regions of interest (ROIs)
    in a set of images, using corresponding mask files to define the ROIs. The function saves a boxplot summarizing
    the distribution of these statistics for each ROI and group of images.

    Steps:
        - Retrieves images and corresponding masks from the specified `image_folder` and `mask_folder` based on
          matching base names (e.g., "image_001_Normalized.png" matches with "image_001_mask.png").
        - For each image-mask pair:
            - Extracts pixel intensities from the image for each ROI as defined by the mask.
            - Computes the standard deviation of pixel intensities for each ROI.
            - Stores the standard deviations for later plotting.
        - Generates a boxplot showing the distribution of standard deviations by ROI, grouped by image type.
        - Saves the plot in the specified `output_folder`.

    Args:
        image_folders (list of str): A list of paths to the folders containing the image files. The filenames of
                                      the images should correspond to those in the mask folder based on the base name.
                                      Example: ["data/for_normalization/Images", "results/Normalized_Images"].
        mask_folder (str): The path to the folder containing mask files. Each mask should label distinct ROIs using
                           integer values (e.g., 0, 1, 2).
        num_rois (int): The number of distinct ROIs in the mask images. Assumes that masks have consistent labeling.
                        For example, `num_rois=11` would mean 11 different regions, labeled 0 through 10.
        output_folder (str): The path to the folder where the resulting boxplot will be saved. The folder will be
                             created if it does not exist.

    Returns:
        None: The function generates a plot and saves it to the specified `output_folder`. It does not return any
              values.
    """

    df = {"group": [],
          "roi": [],
          "std": []}

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    mask_files = glob(os.path.join(mask_folder, "*"))

    for i, mask_file in enumerate(mask_files):
        name = mask_file.split("/")[-1]
        base_name = name.split("_")[0]

        for image_folder in image_folders:

            if image_folder == "data/for_normalization/Images":
                image_name = base_name + ".jpg"
            elif image_folder == "results/Normalized_Images":
                image_name = base_name + "_Normalized.png"
            else:
                image_name = base_name + "_Normalized.png"

            image_file = os.path.join(image_folder, image_name)

            print(f"({i}) {image_file} - {mask_file}")
            image = imread(image_file, as_gray=True)  # Read as grayscale
            mask = imread(mask_file)  # Mask should be integer labeled regions

            for roi in range(num_rois):
                roi_pixels = image[mask == roi]  # Extract pixels for this ROI
                if len(roi_pixels) > 0:
                    std_dev = np.std(roi_pixels)
                    df["roi"].append(roi)
                    df["std"].append(std_dev)

                    if image_folder == "data/for_normalization/Images":
                        df["group"].append("ORG")
                    elif image_folder == "results/Normalized_Images":
                        df["group"].append("JNI")
                    else:
                        df["group"].append("SNI")

    data = pd.DataFrame(df)
    data.to_csv(os.path.join(output_folder, f"std_pi_{num_rois}.csv"), index=False)

    # Plotting
    print("Plotting")
    output_path = os.path.join(output_folder, f"std_pi_{num_rois}.png")
    plt.figure(figsize=(20, 14))
    sns.boxplot(data=df,
                x='roi',
                y='std',
                hue='group',
                showmeans=True,
                meanprops={'marker':'o',
                           'markerfacecolor':'white',
                           'markeredgecolor':'black',
                           'markersize':'8'})
    sns.swarmplot(data=df, x='roi', y='std', hue='group', dodge=True, s=4)
    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title("Standard Deviation of Pixel Intensities by ROI")
    plt.legend(loc=1)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {output_path}")


def plot_he_pi_std(image_folders=["data/for_normalization/Images",
                                  "results/Normalized_Images",
                                  "results/clustering/Normalized_Images"],
                   mask_folder="data/for_normalization/Image_Maps",
                   num_rois=11,
                   output_folder="results/plots"):

    h_df = {"h_group": [], # if the image is ORG, JNI or SNI
            "h_roi": [],
            "h_std": []}

    e_df = {"e_group": [],
            "e_roi": [],
            "e_std": []}

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    mask_files = glob(os.path.join(mask_folder, "*"))

    for i, mask_file in enumerate(mask_files):

        name = mask_file.split("/")[-1]
        base_name = name.split("_")[0]

        for image_folder in image_folders:

            if image_folder == "data/for_normalization/Images":
                image_name = base_name + ".jpg"
            elif image_folder == "results/Normalized_Images":
                image_name = base_name + "_Normalized.png"
            else:
                image_name = base_name + "_Normalized.png"

            image_file = os.path.join(image_folder, image_name)

            print(f"({i}) {image_file} - {mask_file}")

            rgb_image = imread(image_file)
            hed_img = rgb2hed(rgb_image)

            mask = imread(mask_file)  # Mask should be integer labeled regions

            img_h = hed_img[:, :, 0]
            img_e = hed_img[:, :, 1]

            for roi in range(num_rois):

                h_roi_pixels = img_h[mask == roi]  # Extract pixels for this ROI
                e_roi_pixels = img_e[mask == roi]

                if len(h_roi_pixels) > 0:
                    h_df["h_roi"].append(roi)
                    h_std_dev = np.std(h_roi_pixels)
                    h_df["h_std"].append(h_std_dev)

                    if image_folder == "data/for_normalization/Images":
                        h_df["h_group"].append("ORG")
                    elif image_folder == "results/Normalized_Images":
                        h_df["h_group"].append("JNI")
                    else:
                        h_df["h_group"].append("SNI")

                if len(e_roi_pixels) > 0:
                    e_df["e_roi"].append(roi)
                    e_std_dev = np.std(e_roi_pixels)
                    e_df["e_std"].append(e_std_dev)

                    if image_folder == "data/for_normalization/Images":
                        e_df["e_group"].append("ORG")
                    elif image_folder == "results/Normalized_Images":
                        e_df["e_group"].append("JNI")
                    else:
                        e_df["e_group"].append("SNI")

    h_data = pd.DataFrame(h_df)
    e_data = pd.DataFrame(e_df)

    h_data.to_csv(os.path.join(output_folder, f"hematoxilin_pi_std_{num_rois}.csv"), index=False)
    e_data.to_csv(os.path.join(output_folder, f"eosin_pi_std_{num_rois}.csv"), index=False)

    # Plotting Hematoxilin
    print("plotting")
    plt.figure(figsize=(20, 14))
    sns.boxplot(data=h_df,
                x='h_roi',
                y='h_std',
                hue='h_group',
                showmeans=True,
                meanprops={'marker':'o',
                           'markerfacecolor':'white',
                           'markeredgecolor':'black',
                           'markersize':'8'})
    sns.swarmplot(data=h_df, x='h_roi', y='h_std', hue='h_group', dodge=True, s=4)
    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title("Hematoxylin - Standard Deviation of Pixel Intensities by ROI")
    plt.legend(loc=1)
    plt.savefig(os.path.join(output_folder, f"hematoxilin_{num_rois}.png"), dpi=300)
    plt.close()

    print(f"Hematoxilin Plot saved in {output_folder}")

    del h_df, h_data

    # Plotting Eosin
    plt.figure(figsize=(20, 14))
    sns.boxplot(data=e_df,
                x='e_roi',
                y='e_std',
                hue='e_group',
                showmeans=True,
                meanprops={'marker':'o',
                           'markerfacecolor':'white',
                           'markeredgecolor':'black',
                           'markersize':'8'})
    sns.swarmplot(data=e_df, x='e_roi', y='e_std', hue='e_group', dodge=True, s=4)
    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title("Eosin - Standard Deviation of Pixel Intensities by ROI")
    plt.legend(loc=1)
    plt.savefig(os.path.join(output_folder, f"eosin_{num_rois}.png"), dpi=300)
    plt.close()

    print(f"Eosin Plot saved in {output_folder}")



if __name__ == "__main__":

    output_folder = './results/plots'

    compare_pi_std_by_roi("./results/Normalized_Images", './data/for_normalization/KM_Masks/', output_folder, 'JNI_With_Sanyukta_Mask', 8)
    compare_pi_std_by_roi("./results/clustering/Normalized_Images", './data/for_normalization/Image_Maps/', output_folder, 'SNI_With_Jose_Mask', 11)

    imgs_folder = "./data/for_comparison/"
    imgs = os.listdir(imgs_folder)
    org_imgs = [imgs_folder + img for img in imgs]
    norm_with_org_masks = ["./results/JNI/" + img.split(".")[0] + "_Normalized.png" for img in imgs]
    norm_with_our_masks = ["./results/SNI/" + img.split(".")[0] + "_Normalized.png" for img in imgs]

    plot_comparisons(org_imgs, norm_with_org_masks, norm_with_our_masks, output_folder)

    compare_pi_std_by_roi()
    compare_pi_std_by_roi(mask_folder="data/for_normalization/KM_Masks",
                   num_rois=8)

    plot_he_pi_std()
    plot_he_pi_std(mask_folder="data/for_normalization/KM_Masks",
                   num_rois=8)