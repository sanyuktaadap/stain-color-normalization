import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def paired_wilcoxon_per_roi(h_data: pd.DataFrame, e_data: pd.DataFrame, output_csv: str = None):
    """
    Perform paired Wilcoxon signed-rank test (ORG vs SNI) for each ROI and both Hematoxylin and Eosin channels.
    Args:
        h_data (pd.DataFrame): DataFrame for Hematoxylin, must have columns ['hem_group', 'h_roi', 'h_std'].
        e_data (pd.DataFrame): DataFrame for Eosin, must have columns ['eos_group', 'e_roi', 'e_std'].
        output_csv (str, optional): Path to save the output CSV. If None, does not save.
    Returns:
        pd.DataFrame: DataFrame with columns ['ROI', 'H_pvalue', 'E_pvalue'].
    """
    rois = sorted(set(h_data['h_roi']) & set(e_data['e_roi']))
    results = []
    for roi in rois:
        # Hematoxylin
        h_roi = h_data[h_data['h_roi'] == roi]
        h_org = h_roi[h_roi['hem_group'] == 'ORG']['h_std'].values
        h_sni = h_roi[h_roi['hem_group'] == 'SNI']['h_std'].values
        if len(h_org) == len(h_sni) and len(h_org) > 0:
            h_stat, h_p = wilcoxon(h_org, h_sni)
        else:
            h_p = None

        # Eosin
        e_roi = e_data[e_data['e_roi'] == roi]
        e_org = e_roi[e_roi['eos_group'] == 'ORG']['e_std'].values
        e_sni = e_roi[e_roi['eos_group'] == 'SNI']['e_std'].values
        if len(e_org) == len(e_sni) and len(e_org) > 0:
            e_stat, e_p = wilcoxon(e_org, e_sni)
        else:
            e_p = None

        results.append({'ROI': roi, 'H_pvalue': h_p, 'E_pvalue': e_p})

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

# Example usage:
h_data = pd.read_csv('results/plots/csv/hematoxilin_pi_std_8.csv')  # Columns: hem_group, h_roi, h_std
e_data = pd.read_csv('results/plots/csv/eosin_pi_std_8.csv')        # Columns: eos_group, e_roi, e_std
df_pvals = paired_wilcoxon_per_roi(h_data, e_data, output_csv='results/plots/csv/wilcoxon_pvalues.csv')
print(df_pvals)