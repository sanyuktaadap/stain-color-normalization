from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
from tqdm import tqdm
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

# Path to images
maps = "Image_Maps"
images_fold = f"data/for_normalization/{maps}/*"
images_paths = glob.glob(images_fold)

# Initialize a global counter
global_pixel_counts = Counter()

# Loop through all images
for image_path in tqdm(images_paths):
    seg_map = Image.open(image_path).convert("L")  # Convert to grayscale
    seg_array = np.array(seg_map)
    pixel_counts = Counter(seg_array.flatten())
    global_pixel_counts.update(pixel_counts)  # Accumulate

# Sort and prepare data
sorted_labels = sorted(global_pixel_counts.keys())
counts = [global_pixel_counts[label] for label in sorted_labels]

# Save counts to CSV
df = pd.DataFrame({
    'Region Label': sorted_labels,
    'Pixel Count': counts
})
os.makedirs("results/csvs", exist_ok=True)
df.to_csv(f"results/plots/csv/pixel_count_stats_combined_{maps}.csv", index=False)

# Plotting
plt.figure(figsize=(14, 6))
bars = plt.bar(sorted_labels, counts, color='mediumpurple')

# Add scientific notation count on top of each bar
for bar, count in zip(bars, counts):
    height = bar.get_height()
    sci_notation = f"{count:.1e}"
    plt.text(bar.get_x() + bar.get_width() / 2,
             height,
             sci_notation,
             ha='center',
             va='bottom',
             fontsize=8)

# Format axes
plt.xlabel("Region Label")
plt.ylabel("Total Pixel Count")
plt.title(f"Aggregated Pixel Count per Region")
plt.xticks(ticks=sorted_labels, labels=sorted_labels, ha='right')

# Use scientific notation on y-axis too
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Create results directory if it doesn't exist
os.makedirs("results/plots", exist_ok=True)
plt.savefig(f"results/plots/pixel_count_stats_combined_{maps}.png", dpi=300)