import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def paired_wilcoxon_per_roi(data1: pd.DataFrame, data2: pd.DataFrame, output_csv: str = None):
    """
    Perform paired Wilcoxon signed-rank test (JNI vs SNI) for each ROI and Hematoxylin channels of both cohorts.
    Args:
        data1 (pd.DataFrame): DataFrame for Hematoxylin, must have columns ['group', 'roi', 'std'].
        data2 (pd.DataFrame): DataFrame for Hematoxylin, must have columns ['group', 'roi', 'std'].
        output_csv (str, optional): Path to save the output CSV. If None, does not save.
    Returns:
        pd.DataFrame: DataFrame with columns ['ROI', 'INSM', 'INUM'].
    """
    rois = sorted(set(data1['roi']) & set(data2['roi']))
    results = []
    for roi in rois:
        # Hematoxylin
        h_roi = data1[data1['roi'] == roi]
        h_jni = h_roi[h_roi['group'] == 'INSM']['std'].values
        h_sni = h_roi[h_roi['group'] == 'INUM']['std'].values
        if len(h_jni) == len(h_sni) and len(h_jni) > 0:
            h_stat, h_p = wilcoxon(h_jni, h_sni)
        else:
            h_p = None

        # Eosin
        e_roi = data2[data2['roi'] == roi]
        e_jni = e_roi[e_roi['group'] == 'INSM']['std'].values
        e_sni = e_roi[e_roi['group'] == 'INUM']['std'].values
        if len(e_jni) == len(e_sni) and len(e_jni) > 0:
            e_stat, e_p = wilcoxon(e_jni, e_sni)
        else:
            e_p = None

        results.append({'ROI': roi, 'INSM': h_p, 'INUM': e_p})

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

# Example usage:
h1_data = pd.read_csv('results/plots/csv/hematoxilin_pi_std_11.csv')  # Columns: hem_group, h_roi, h_std
h2_data = pd.read_csv('results/plots/csv/hematoxilin_pi_std_8.csv') # Columns: hem_group, h_roi, h_std
e1_data = pd.read_csv('results/plots/csv/eosin_pi_std_11.csv')        # Columns: eos_group, e_roi, e_std
e2_data = pd.read_csv('results/plots/csv/eosin_pi_std_8.csv')       # Columns: eos_group, e_roi, e_std
df_pvals = paired_wilcoxon_per_roi(h1_data, h2_data, output_csv='results/plots/wilcoxon_pvals_JNIvsSNI_hem.csv')
df_pvals = paired_wilcoxon_per_roi(e1_data, e2_data, output_csv='results/plots/wilcoxon_pvals_JNIvsSNI_eos.csv')