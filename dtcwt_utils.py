import numpy as np
import subprocess
import pickle
import os
import time
import signal


def run_dtcwt(signal, t, nlevels=5, venv_path=None, timeout=30):
    """
    Run DTCWT decomposition using a subprocess to avoid numpy version conflicts.

    Args:
        signal (np.ndarray): Input signal array.
        t (np.ndarray): Time array corresponding to the signal.
        nlevels (int): Number of decomposition levels.
        venv_path (str): Path to the virtual environment to use for subprocess.
        timeout (int): Maximum time in seconds for subprocess execution (default: 30).

    Returns:
        list: List of highpass coefficients for each level.
    """
    if len(signal) != len(t):
        raise ValueError(
            f"Signal length ({len(signal)}) does not match time array length ({len(t)})"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or inf values")

    # Ensure signal length is a power of 2 for DTCWT
    target_length = 2 ** int(np.ceil(np.log2(len(signal))))
    print(f"Original signal length: {len(signal)}")
    print(f"Target length: {target_length}")
    signal_padded = np.pad(signal, (0, target_length - len(signal)), mode="constant")
    t_padded = np.pad(t, (0, target_length - len(t)), mode="edge")

    # Prepare input data for subprocess
    input_data = {"signal": signal_padded, "t": t_padded, "nlevels": nlevels}

    # Serialize input data
    input_pickle = pickle.dumps(input_data)

    # Determine the Python executable to use
    if venv_path and os.path.exists(os.path.join(venv_path, "bin", "python")):
        python_exec = os.path.join(venv_path, "bin", "python")
    else:
        python_exec = "python"

    print(f"Using Python executable: {python_exec}")

    # Run the DTCWT processing script in a subprocess with piped input/output
    process = None
    try:
        start_time = time.time()
        process = subprocess.Popen(
            [python_exec, "-u", "dtcwt_processing.py"],  # -u for unbuffered output
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Handle bytes directly
            preexec_fn=(
                os.setsid if os.name != "nt" else None
            ),  # Enable process group on Unix
        )
        stdout, stderr = process.communicate(input=input_pickle, timeout=timeout)
        if process.returncode != 0:
            print(f"DTCWT subprocess error details: {stderr.decode('utf-8')}")
            raise ValueError(
                f"DTCWT subprocess failed with exit code {process.returncode}: {stderr.decode('utf-8')}"
            )
        print(f"DTCWT subprocess stderr: {stderr.decode('utf-8')}")
        print(f"DTCWT subprocess completed in {time.time() - start_time:.2f} seconds")
        # Parse the binary output
        output = pickle.loads(stdout)
        coeffs = output["highpass_coeffs"]
    except subprocess.TimeoutExpired:
        print(f"DTCWT subprocess timed out after {timeout} seconds")
        if process:
            # Forcefully terminate the process group
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        raise ValueError(f"DTCWT subprocess timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        print(f"DTCWT subprocess error details: {e.stderr.decode('utf-8')}")
        raise ValueError(
            f"DTCWT subprocess failed with exit code {e.returncode}: {e.stderr.decode('utf-8')}"
        )
    except Exception as e:
        print(f"DTCWT processing error: {str(e)}")
        raise
    finally:
        # Ensure process is terminated
        if process and process.poll() is None:
            process.kill()

    return coeffs
