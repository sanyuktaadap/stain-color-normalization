import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_std_from_csv(output_folder="results/plots", num_rois=11):
    # Load CSVs
    h_csv_path = os.path.join(output_folder, f"csv/hematoxilin_pi_std_{num_rois}.csv")
    e_csv_path = os.path.join(output_folder, f"csv/eosin_pi_std_{num_rois}.csv")

    h_data = pd.read_csv(h_csv_path)
    e_data = pd.read_csv(e_csv_path)

    # Set global font size
    plt.rcParams.update({'font.size': 32})

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(32, 14), sharey=True)

    # Hematoxylin Plot
    sns.boxplot(data=h_data, x='h_roi', y='h_std', hue='hem_group',
                showmeans=True, ax=axs[0],
                linewidth=2,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 8})
    sns.swarmplot(data=h_data, x='h_roi', y='h_std', hue='hem_group',
                  dodge=True, s=4, ax=axs[0], palette=sns.color_palette(n_colors=3))
    axs[0].set_title("Hematoxylin")
    axs[0].set_xlabel("Regions of Interest")
    axs[0].set_ylabel("Std. Dev. of Pixel Intensities")
    axs[0].legend_.remove()

    # Eosin Plot
    sns.boxplot(data=e_data, x='e_roi', y='e_std', hue='eos_group',
                showmeans=True, ax=axs[1],
                linewidth=2,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 8})
    sns.swarmplot(data=e_data, x='e_roi', y='e_std', hue='eos_group',
                  dodge=True, s=4, ax=axs[1], palette=sns.color_palette(n_colors=3))
    axs[1].set_title("Eosin")
    axs[1].set_xlabel("Regions of Interest")
    axs[1].set_ylabel("")
    axs[1].legend_.remove()

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_folder, f"hed_std_pi_{num_rois}.png"), dpi=300)
    plt.close()

    # # Individual plots
    # # Hematoxylin
    # plt.figure(figsize=(20, 14))
    # sns.boxplot(data=h_data, x='h_roi', y='h_std', hue='hem_group', showmeans=True,
    #             meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 8})
    # sns.swarmplot(data=h_data, x='h_roi', y='h_std', hue='hem_group', dodge=True, s=4)
    # plt.xlabel("Regions of Interest")
    # plt.ylabel("Standard Deviation of Pixel Intensities")
    # plt.title("Hematoxylin - Standard Deviation of Pixel Intensities by ROI")
    # plt.legend(loc=1)
    # plt.savefig(os.path.join(output_folder, f"hematoxilin_{num_rois}_from_csv.png"), dpi=300)
    # plt.close()

    # # Eosin
    # plt.figure(figsize=(20, 14))
    # sns.boxplot(data=e_data, x='e_roi', y='e_std', hue='eos_group', showmeans=True,
    #             meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 8})
    # sns.swarmplot(data=e_data, x='e_roi', y='e_std', hue='eos_group', dodge=True, s=4)
    # plt.xlabel("Regions of Interest")
    # plt.ylabel("Standard Deviation of Pixel Intensities")
    # plt.title("Eosin - Standard Deviation of Pixel Intensities by ROI")
    # plt.legend(loc=1)
    # plt.savefig(os.path.join(output_folder, f"eosin_{num_rois}_from_csv.png"), dpi=300)
    # plt.close()

    # print(f"Plots from CSV saved in {output_folder}")

# Example usage
plot_std_from_csv(num_rois=11)
plot_std_from_csv(output_folder="results/plots", num_rois=8)