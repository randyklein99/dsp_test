# dtcwt_utils.py
import subprocess
import pickle
import os

def run_dtcwt(signal, t, dtcwt_script="dtcwt_processing.py", venv_path="./.venv_dtcwt"):
    """
    Run DTCWT on a signal via subprocess and return coefficients.

    Args:
        signal (np.ndarray): Input signal array.
        t (np.ndarray): Time array for the signal.
        dtcwt_script (str): Path to the DTCWT processing script.
        venv_path (str): Path to the dtcwt virtual environment.

    Returns:
        np.ndarray: Magnitude of level 1 highpass coefficients.

    Raises:
        ValueError: If subprocess fails or returns no data.
    """
    dtcwt_python = os.path.abspath(os.path.join(venv_path, "bin/python"))
    process = subprocess.Popen(
        [dtcwt_python, dtcwt_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    signal_data = pickle.dumps({"t": t, "signal": signal})
    stdout, stderr = process.communicate(input=signal_data)

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise ValueError(f"DTCWT subprocess failed with exit code {process.returncode}: {error_msg}")
    if stderr:
        print("DTCWT messages/errors:", stderr.decode())
    if not stdout:
        raise ValueError("No data received from dtcwt_processing.py")

    results = pickle.loads(stdout)
    return results["coeffs"].flatten()