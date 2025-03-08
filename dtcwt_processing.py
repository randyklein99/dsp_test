import pickle
import sys
import numpy as np
from dtcwt import Transform1d

# Read serialized data from stdin
try:
    # Load input data from stdin
    input_data = pickle.load(sys.stdin.buffer)
    signal = input_data["signal"]
    t = input_data["t"]
    nlevels = input_data["nlevels"]
    print(f"Starting DTCWT processing...", file=sys.stderr)
    print(f"Signal length: {len(signal)}, nlevels: {nlevels}", file=sys.stderr)

    # Perform DTCWT
    transform = Transform1d()
    coeffs = transform.forward(signal, nlevels=nlevels)

    # Extract highpass coefficients, handling different DTCWT library versions
    highpass_coeffs = []
    lowpass_coeff = None

    # Try different access methods based on DTCWT version
    if hasattr(coeffs, "highpasses"):
        # Newer DTCWT version
        highpass_coeffs = [np.array(c) for c in coeffs.highpasses]
        # Try to get lowpass coefficient if available
        if hasattr(coeffs, "lowpasses"):
            lowpass_coeff = np.array(coeffs.lowpasses[-1])  # Last lowpass coefficient
        else:
            # If no lowpasses attribute, try other possible attribute names
            if hasattr(coeffs, "lowpass"):
                lowpass_coeff = np.array(coeffs.lowpass)
            else:
                # Create a dummy lowpass coefficient
                lowpass_coeff = np.zeros(len(signal) // (2**nlevels), dtype=complex)
    else:
        # Older DTCWT version might structure results differently
        # Try to interpret coeffs as a tuple (Yl, Yh) as in some versions
        if isinstance(coeffs, tuple) and len(coeffs) == 2:
            lowpass_coeff = np.array(coeffs[0])  # Yl is the lowpass
            highpass_coeffs = [
                np.array(c) for c in coeffs[1]
            ]  # Yh is list of highpass bands
        else:
            # Last resort: assume coeffs itself is the list of coefficients
            highpass_coeffs = [np.array(coeffs)]
            lowpass_coeff = np.zeros(len(signal) // (2**nlevels), dtype=complex)

    # Ensure we have some highpass coefficients
    if not highpass_coeffs:
        raise ValueError("Could not extract highpass coefficients from DTCWT output")

    # Make sure we're only sending back NumPy arrays that can be pickled
    output = {
        "highpass_coeffs": [
            hp.tolist() if isinstance(hp, np.ndarray) else list(hp)
            for hp in highpass_coeffs
        ],
        "lowpass_coeff": (
            lowpass_coeff.tolist()
            if isinstance(lowpass_coeff, np.ndarray)
            else list(lowpass_coeff)
        ),
    }
    print("Computed DTCWT coefficients successfully", file=sys.stderr)

    # Serialize to stdout without any print statements before it
    sys.stdout.buffer.write(pickle.dumps(output))
    sys.stdout.buffer.flush()
    sys.exit(0)

except Exception as e:
    print(f"Error in dtcwt_processing: {str(e)}", file=sys.stderr)
    sys.exit(1)
