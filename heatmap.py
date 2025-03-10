# %%
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor import extract_features
from signal_generator import generate_80211ag_preamble

# Generate a single preamble signal (as in your current int_extract.py)
fs = 20e6  # Sampling frequency
t = np.arange(0, 24e-6, 1/fs)
add_rf_fingerprint = True
t, preamble = generate_80211ag_preamble(fs, add_rf_fingerprint=add_rf_fingerprint)

# Extract features
feature_vector = extract_features(preamble, t, fs)

# Reshape and normalize the feature vector
# 135 elements = 15 segments x 9 stats
num_segments = 15
num_stats = 9
num_markers = 1  # Single burst for now
feature_matrix = feature_vector.reshape(num_markers, num_segments, num_stats)

# Average across stats to get 15 segments per marker
feature_matrix = np.mean(feature_matrix, axis=2)  # Shape: (1, 15)

# Normalize to 0-1 range
feature_matrix = (feature_matrix - np.min(feature_matrix)) / (np.max(feature_matrix) - np.min(feature_matrix))

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(feature_matrix, cmap='jet', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Normalized Feature Value')
plt.title('WD Fingerprint (Single Device)')
plt.xlabel('Segment (1-15)')
plt.ylabel('DNA Marker')
plt.yticks([0], ['Marker 1'])
plt.xticks(np.arange(15), np.arange(1, 16))
plt.show()
# %%
