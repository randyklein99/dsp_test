import numpy as np
import pickle
from dtcwt import Transform1d
import sys

try:
    # Read signal from stdin
    signal_data = sys.stdin.buffer.read()
    if not signal_data:
        print("Error: No input data received", file=sys.stderr)
        sys.exit(1)
    data = pickle.loads(signal_data)
    t = data["t"]
    signal = data["signal"]
    print(f"Received signal length: {len(signal)}", file=sys.stderr)

    # Perform DTCWT
    transform = Transform1d()
    coeffs = transform.forward(signal, nlevels=4)
    mag = np.abs(coeffs.highpasses[0])
    print("DTCWT env NumPy version:", np.__version__, file=sys.stderr)
    print(f"Computed coefficients length: {len(mag)}", file=sys.stderr)

    # Send results back via stdout
    results = pickle.dumps({"coeffs": mag})
    sys.stdout.buffer.write(results)
    sys.stdout.flush()
except Exception as e:
    print(f"Error in dtcwt_processing: {str(e)}", file=sys.stderr)
    sys.exit(1)