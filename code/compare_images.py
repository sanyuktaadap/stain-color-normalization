import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob
from PIL import Image
import seaborn as sns
from skimage.color import rgb2hed

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
                                     figsize=(15, 60))

    fig_rgb.suptitle('Comparison of Images in RGB Color Space', fontsize=24)

    # Set column titles
    column_titles = ["Original", "Normalized With Original Mask", "Normalized Our Mask"]
    for col, title in enumerate(column_titles):
        axes_rgb[0, col].set_title(title, fontsize=24, pad=20)

    for i in range(num_images):
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
    fig_rgb.savefig(rgb_output_path, dpi=300)
    plt.close(fig_rgb)

    print(f"RGB comparison plot saved at: {rgb_output_path}")


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
          base name (e.g., "image_001_Normalized.png" matches with "image_001_mask.png").
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
        base_name = name.split("_")[0]
        image_name = base_name + "_Normalized.png"
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

    # Plotting Hematoxilin
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=h_df, x='h_roi', y='h_std', hue='h_group')
    sns.swarmplot(data=h_df, x='h_roi', y='h_std', hue='h_group', dodge=True, s=5)
    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title("Hematoxylin - Standard Deviation of Pixel Intensities by ROI")
    plt.savefig(os.path.join(output_folder, f"hematoxilin_{num_rois}.png"), dpi=300)
    plt.close()

    # Plotting Eosin
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=e_df, x='e_roi', y='e_std', hue='e_group')
    sns.swarmplot(data=e_df, x='e_roi', y='e_std', hue='e_group', dodge=True, s=5)
    plt.xlabel("Regions of Interest")
    plt.ylabel("Standard Deviation of Pixel Intensities")
    plt.title("Eosin - Standard Deviation of Pixel Intensities by ROI")
    plt.savefig(os.path.join(output_folder, f"eosin_{num_rois}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":

    output_folder = './results/plots'

    compare_pi_std_by_roi("./results/Normalized_Images", './data/for_normalization/KM_Masks/', output_folder, 'JNI_With_Sanyukta_Mask', 8)
    compare_pi_std_by_roi("./results/clustering/Normalized_Images", './data/for_normalization/Image_Maps/', output_folder, 'SNI_With_Jose_Mask', 11)

    imgs_folder = "./data/for_comparison/Original/"
    imgs = os.listdir(imgs_folder)
    base_names = [img.split("/")[-1] for img in imgs]
    org_imgs = [imgs_folder + img for img in base_names]
    norm_with_org_masks = ["./results/Normalized_Images/" + img.split(".")[0] + "_Normalized.png" for img in base_names]
    norm_with_our_masks = ["./results/clustering/Normalized_Images/" + img.split(".")[0] + "_Normalized.png" for img in base_names]

    plot_comparisons(org_imgs, norm_with_org_masks, norm_with_our_masks, output_folder)

    plot_he_pi_std()
    plot_he_pi_std(mask_folder="data/for_normalization/KM_Masks",
                   num_rois=8)