import scipy.io as sio
import numpy as np

# Specify the path to your .mat file
mat_file_path = "/mnt/i/randyHomeCenturion/randy/Documents/PhD/Dissertation/Matlab/Chamber/Cisco/Originals/N4U9_Composite.mat"  # Replace with your actual file path
mat_data = sio.loadmat(mat_file_path)
print("File loaded successfully!")

# Extract variables
xdelta = mat_data["XDelta"]
freq_max = mat_data["FreqValidMax"]
freq_min = mat_data["FreqValidMin"]
signal = mat_data["Signal"]
state_i = mat_data["StateI"]
state_q = mat_data["StateQ"]

# Save to .npz (compressed NumPy archive)
# output_file = "/mnt/i/randyHomeCenturion/randy/Documents/PhD/Dissertation/Matlab/Chamber/Uncompressed/rf_data.npz"
output_file = "./data/rf_data.npz"

np.savez_compressed(
    output_file,
    XDelta=xdelta,
    FreqValidMax=freq_max,
    FreqValidMin=freq_min,
    Signal=signal,
    StateI=state_i,
    StateQ=state_q,
)
print(f"Data saved to {output_file}")

# Optional: Verify by loading it back
loaded_data = np.load(output_file)
print("Loaded variables:", list(loaded_data.keys()))
print("Signal shape (verification):", loaded_data["Signal"].shape)
