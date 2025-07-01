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

    # Filter out ROI 0 (background)
    h_data = h_data[h_data['h_roi'] != 0]
    e_data = e_data[e_data['e_roi'] != 0]

    # Set global font size
    plt.rcParams.update({'font.size': 33})

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


def save_iqr_tables(output_folder="results/plots", num_rois=11):
    # Load CSVs
    h_csv_path = os.path.join(output_folder, f"hematoxilin_pi_std_{num_rois}.csv")
    e_csv_path = os.path.join(output_folder, f"eosin_pi_std_{num_rois}.csv")

    h_data = pd.read_csv(h_csv_path)
    e_data = pd.read_csv(e_csv_path)

    # Filter out ROI 0 (background)
    h_data = h_data[h_data['h_roi'] != 0]
    e_data = e_data[e_data['e_roi'] != 0]

    # Compute IQR and Median for Hematoxylin
    h_iqr = h_data.groupby(['h_roi', 'hem_group'])['h_std'].agg(
        Q1=lambda x: x.quantile(0.25),
        Median='median',
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()
    h_iqr['IQR'] = h_iqr['Q3'] - h_iqr['Q1']

    # Compute IQR and Median for Eosin
    e_iqr = e_data.groupby(['e_roi', 'eos_group'])['e_std'].agg(
        Q1=lambda x: x.quantile(0.25),
        Median='median',
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()
    e_iqr['IQR'] = e_iqr['Q3'] - e_iqr['Q1']

    # Save tables
    iqr_folder = os.path.join(output_folder, "iqr_tables")
    os.makedirs(iqr_folder, exist_ok=True)
    h_iqr.to_csv(os.path.join(iqr_folder, f"hematoxilin_iqr_{num_rois}.csv"), index=False)
    e_iqr.to_csv(os.path.join(iqr_folder, f"eosin_iqr_{num_rois}.csv"), index=False)

    print(f"IQR + median tables saved in: {iqr_folder}")


def detect_stain_columns(df):
    """Detects whether data is for Hematoxylin or Eosin and returns standardized column mapping."""
    if 'h_roi' in df.columns:
        return {
            'roi': 'h_roi',
            'group': 'hem_group',
            'median': 'Median',
            'q1': 'Q1',
            'q3': 'Q3',
            'iqr': 'IQR',
            'stain': 'Hematoxylin'
        }
    elif 'e_roi' in df.columns:
        return {
            'roi': 'e_roi',
            'group': 'eos_group',
            'median': 'Median',
            'q1': 'Q1',
            'q3': 'Q3',
            'iqr': 'IQR',
            'stain': 'Eosin'
        }
    else:
        raise ValueError("CSV must contain either 'h_roi' or 'e_roi' columns.")

def plot_median_barplot(csv_path, output_folder):

    df = pd.read_csv(csv_path)
    col = detect_stain_columns(df)
    if str(11) in csv_path:
        num_rois = 11
    else:
        num_rois = 8

    group_order = ['ORG', 'JNI', 'SNI']
    df[col['group']] = pd.Categorical(df[col['group']], categories=group_order, ordered=True)
    df = df.sort_values(by=[col['roi'], col['group']])

    os.makedirs(os.path.join(output_folder, "iqr"), exist_ok=True)
    save_path = os.path.join(output_folder, "iqr", f"median_barplot_{col['stain'].lower()}_roi{num_rois}.png")

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x=col['roi'], y=col['median'], hue=col['group'], hue_order=group_order)
    plt.title(f"Median Std. Dev. of Pixel Intensities per ROI ({col['stain']})")
    plt.xlabel("ROI")
    plt.ylabel("Median Std. Dev.")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_median_heatmap(csv_path, output_folder):

    df = pd.read_csv(csv_path)
    col = detect_stain_columns(df)
    if str(11) in csv_path:
        num_rois = 11
    else:
        num_rois = 8

    group_order = ['ORG', 'JNI', 'SNI']
    pivot_median = df.pivot(index=col['roi'], columns=col['group'], values=col['median'])
    pivot_median = pivot_median[group_order]  # Reorder x-axis
    pivot_median = pivot_median.sort_index()  # Sort ROI numerically (y-axis)

    os.makedirs(os.path.join(output_folder, "iqr"), exist_ok=True)
    save_path = os.path.join(output_folder, "iqr", f"median_heatmap_{col['stain'].lower()}_roi{num_rois}.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_median, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=0.5)
    plt.title(f"Median Std. Dev. per ROI and Group ({col['stain']})")
    plt.xlabel("Group")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_iqr_heatmap(csv_path, output_folder):

    df = pd.read_csv(csv_path)
    col = detect_stain_columns(df)
    if str(11) in csv_path:
        num_rois = 11
    else:
        num_rois = 8

    os.makedirs(os.path.join(output_folder, "iqr"), exist_ok=True)
    save_path = os.path.join(output_folder, "iqr", f"iqr_heatmap_{col['stain'].lower()}_roi{num_rois}.png")

    # Pivot with specified group order and sorted ROI
    group_order = ['ORG', 'JNI', 'SNI']
    pivot_iqr = df.pivot(index=col['roi'], columns=col['group'], values=col['iqr'])
    pivot_iqr = pivot_iqr[group_order]  # Set column order
    pivot_iqr = pivot_iqr.sort_index()  # Sort rows by ROI

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_iqr, annot=True, cmap='OrRd', fmt=".4f", linewidths=0.5)
    plt.title(f"IQR of Std. Dev. per ROI and Group ({col['stain']})")
    plt.xlabel("Group")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



# Example usage
plot_std_from_csv(num_rois=11)
plot_std_from_csv(output_folder="results/plots", num_rois=8)

save_iqr_tables(output_folder="results/plots/csv/", num_rois=11)
save_iqr_tables(output_folder="results/plots/csv/", num_rois=8)

path = "results/plots/csv/iqr_tables"
output_folder="results/plots"
iqr_tables = os.listdir(path)

for iqr_table in iqr_tables:
    tab_path = os.path.join(path, iqr_table)
    plot_median_barplot(tab_path,output_folder)
    plot_median_heatmap(tab_path,output_folder)
    plot_iqr_heatmap(tab_path,output_folder)