# dtcwt_processing.py
import numpy as np
import pickle
from dtcwt import Transform1d
import sys

try:
    # Read signal from stdin
    signal_data = sys.stdin.buffer.read()
    print(f"Raw input (first 50 bytes): {signal_data[:50]}", file=sys.stderr)  # Debug
    if not signal_data:
        print("Error: No input data received", file=sys.stderr)
        sys.exit(1)
    data = pickle.loads(signal_data)
    t = data["t"]
    signal = np.array(data["signal"])
    print(f"Received signal length: {len(signal)}", file=sys.stderr)

    # Pad signal to even length if necessary
    if len(signal) % 2 != 0:
        padding = np.zeros(1)
        signal = np.concatenate((signal, padding))

    # Perform DTCWT
    transform = Transform1d()
    coeffs = transform.forward(signal, nlevels=4)
    print("DTCWT env NumPy version:", np.__version__, file=sys.stderr)
    print(f"Computed coefficients length (highpass level 0): {len(coeffs.highpasses[0])}", file=sys.stderr)

    # Convert coeffs to a simple NumPy array (e.g., flatten highpass coefficients)
    highpass_coeffs = np.concatenate([coeffs.highpasses[i].flatten() for i in range(4)])
    results = pickle.dumps({"coeffs": highpass_coeffs})
    sys.stdout.buffer.write(results)
    sys.stdout.flush()
except Exception as e:
    print(f"Error in dtcwt_processing: {str(e)}", file=sys.stderr)
    sys.exit(1)