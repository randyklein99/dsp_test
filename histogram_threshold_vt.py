# %%
# histogram_threshold_vt.py
import numpy as np
import matplotlib.pyplot as plt
import os
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, analyze_noise
from denoising import denoise_signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Enable interactive mode
plt.ion()

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define parameters
fs = 20e6  # Sampling frequency in Hz
base_seed = 42  # Base seed for reproducibility
num_simulations = 50  # Reduced for faster iterations
debug_level = 0  # Disable debug prints
window_size = 320  # Fixed at 320 samples (16 µs)
snr_db_range = np.concatenate([np.arange(-3, 12, 1), np.arange(12, 31, 6)])  # -3 to 11 dB (1 dB), 12 to 30 dB (6 dB)
snr_low_db_range = np.arange(-3, 12, 1)  # Low SNR range for metric (-3 to 11 dB)
noise_duration = 60e-6  # Noise region duration (60 µs)
noise_samples = int(noise_duration * fs)
preamble_duration = 20e-6  # Approximate preamble duration
threshold_multiplier = 0.1  # Reduced to preserve signal

def calculate_snr(signal_power, noise_power):
    """Calculate SNR in dB given signal and noise power."""
    return 10 * np.log10(signal_power / noise_power)

def add_noise_to_signal(signal, snr_db, fs):
    """Add noise to signal to achieve specified SNR, return noisy signal and noise std."""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * noise_std / np.sqrt(2)
    noisy_signal = signal + noise
    return noisy_signal, noise_std

def compute_variance_trajectory(signal, t, window_size, end_idx=None):
    """Compute variance trajectory up to a specified end index."""
    variances = []
    end_idx = len(signal) - window_size + 1 if end_idx is None else min(end_idx, len(signal) - window_size + 1)
    for i in range(end_idx):
        window = signal[i:i + window_size]
        variance = np.var(np.abs(window))
        variances.append(variance)
    return np.array(variances)

def generate_noise_and_preamble(snr_db, seed=None):
    """Generate noise and preamble separately, applying RF fingerprint only to preamble."""
    np.random.seed(seed)
    # Generate noise region
    noise = (np.random.randn(noise_samples) + 1j * np.random.randn(noise_samples)) * np.sqrt(0.1) / np.sqrt(2)  # Approx 0 dB noise
    t_noise = np.arange(noise_samples) / fs
    
    # Generate preamble with RF fingerprint
    t_preamble, preamble_clean = generate_80211ag_preamble(fs=fs, add_rf_fingerprint=True, seed=seed, debug_level=debug_level)
    preamble_samples = int(preamble_duration * fs)
    preamble_clean = preamble_clean[:preamble_samples]  # Truncate to match noise duration if needed
    noisy_preamble, _ = add_noise_to_signal(preamble_clean, snr_db, fs)
    
    # Combine signals
    input_signal = np.concatenate([noise, noisy_preamble])
    t = np.concatenate([t_noise, t_noise[-1] + (1 / fs) + t_preamble[:preamble_samples]])
    return input_signal, t

def evaluate_threshold_method(input_signal, t, snr_db, threshold_raw, threshold_denoised):
    """Evaluate using separate thresholds for Raw VT and Denoised VT."""
    effective_window_size = min(window_size, len(input_signal) - 1)
    
    # Raw VT analysis
    noise_stats_raw = analyze_noise(input_signal, t, window_size=effective_window_size, 
                                   noise_region=(0, noise_duration), debug_level=debug_level)
    variance_traj_raw, _, _, trigger_idx_raw = variance_trajectory_detector(
        input_signal, t, window_size=effective_window_size, threshold=threshold_raw, debug_level=debug_level
    )
    # Find the first index where variance exceeds threshold significantly
    trigger_idx_raw_full = -1
    for i in range(len(variance_traj_raw)):
        if variance_traj_raw[i] > threshold_raw * 2.0:  # Use 2.0x threshold to catch the rise
            trigger_idx_raw_full = i + effective_window_size - 1
            break
    error_raw = trigger_idx_raw_full - noise_samples if trigger_idx_raw_full >= 0 else float('inf')
    success_raw = abs(error_raw) <= int(0.5e-6 * fs) if error_raw != float('inf') else False  # Tighten error margin to 0.5 µs
    
    # Denoised VT analysis
    denoised_signal = denoise_signal(input_signal, t, threshold_multiplier=threshold_multiplier, debug_level=debug_level)
    noise_stats_denoised = analyze_noise(denoised_signal, t, window_size=effective_window_size, 
                                        noise_region=(0, noise_duration), debug_level=debug_level)
    variance_traj_denoised, _, _, trigger_idx_denoised = variance_trajectory_detector(
        denoised_signal, t, window_size=effective_window_size, threshold=threshold_denoised, debug_level=debug_level
    )
    # Find the first index where variance exceeds threshold significantly
    trigger_idx_denoised_full = -1
    for i in range(len(variance_traj_denoised)):
        if variance_traj_denoised[i] > threshold_denoised * 2.0:  # Use 2.0x threshold to catch the rise
            trigger_idx_denoised_full = i + effective_window_size - 1
            break
    error_denoised = trigger_idx_denoised_full - noise_samples if trigger_idx_denoised_full >= 0 else float('inf')
    success_denoised = abs(error_denoised) <= int(0.5e-6 * fs) if error_denoised != float('inf') else False  # Tighten error margin to 0.5 µs
    
    # Plot VT over time for debugging
    t_vt = np.arange(len(variance_traj_raw)) * (window_size / fs)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_vt, variance_traj_raw, 'b-', label='Raw VT')
    ax.plot(t_vt, variance_traj_denoised, 'g-', label='Denoised VT')
    ax.axvline(noise_duration, color='r', linestyle='--', label='Noise-Preamble Boundary')
    ax.axhline(threshold_raw, color='b', linestyle='--', label=f'Raw VT Threshold ({threshold_raw:.4f})')
    ax.axhline(threshold_denoised, color='g', linestyle='--', label=f'Denoised VT Threshold ({threshold_denoised:.4f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Trajectory vs Time')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(plots_dir, 'vt_time_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return error_raw, error_denoised, success_raw, success_denoised, variance_traj_raw, variance_traj_denoised

def run_simulation_for_roc(sim):
    """Run a single simulation to collect VT values for ROC curve at 6 dB SNR."""
    np.random.seed(base_seed + sim)
    
    input_signal, t = generate_noise_and_preamble(6)  # 6 dB SNR
    
    # Compute VT for raw and denoised signals
    vt_raw = compute_variance_trajectory(input_signal, t, window_size)
    denoised_signal = denoise_signal(input_signal, t, threshold_multiplier=threshold_multiplier, debug_level=debug_level)
    vt_denoised = compute_variance_trajectory(denoised_signal, t, window_size)
    
    # Label VT values: 0 for noise (0–60 µs), 1 for preamble (60 µs onwards)
    noise_region_end_idx = int(noise_samples / window_size) - 1  # Adjust for window overlap
    labels = np.zeros(len(vt_raw))
    labels[noise_region_end_idx + 1:] = 1  # Preamble region, shifted to avoid overlap
    
    return vt_raw, vt_denoised, labels

def run_simulation_for_histogram(sim):
    """Run a single simulation to collect VT values for histogram generation at 0 dB SNR."""
    np.random.seed(base_seed + sim)
    
    input_signal, t = generate_noise_and_preamble(0)  # 0 dB SNR for noise-dominated region
    
    # Compute VT for raw and denoised signals, limiting to noise region
    noise_region_end_idx = noise_samples - window_size + 1  # Stop before window overlaps with preamble
    vt_raw = compute_variance_trajectory(input_signal, t, window_size, end_idx=noise_region_end_idx)
    denoised_signal = denoise_signal(input_signal, t, threshold_multiplier=threshold_multiplier, debug_level=debug_level)
    vt_denoised = compute_variance_trajectory(denoised_signal, t, window_size, end_idx=noise_region_end_idx)
    
    return vt_raw, vt_denoised

def run_simulation(sim, threshold_raw, threshold_denoised):
    """Run a single Monte Carlo simulation comparing Raw VT and Denoised VT."""
    np.random.seed(base_seed + sim)
    
    input_signal, t = generate_noise_and_preamble(0, seed=base_seed + sim)  # Use 0 dB for main simulations
    
    sim_results = {'errors_raw': [], 'errors_denoised': [], 'success_raw': [], 'success_denoised': []}
    
    for snr_db in snr_db_range:
        noisy_signal, _ = add_noise_to_signal(input_signal[noise_samples:], snr_db, fs)  # Add noise to preamble only
        input_signal_with_noise = np.concatenate([input_signal[:noise_samples], noisy_signal])
        t_with_noise = np.concatenate([t[:noise_samples], t[noise_samples-1] + (1 / fs) + t[noise_samples:]])
        
        error_raw, error_denoised, success_raw, success_denoised, _, _ = evaluate_threshold_method(
            input_signal_with_noise, t_with_noise, snr_db, threshold_raw, threshold_denoised
        )
        sim_results['errors_raw'].append(error_raw)
        sim_results['errors_denoised'].append(error_denoised)
        sim_results['success_raw'].append(success_raw)
        sim_results['success_denoised'].append(success_denoised)
    
    return sim_results

def generate_roc_curve(vt_values, labels, title):
    """Generate ROC curve for given VT values and labels."""
    thresholds = np.linspace(min(vt_values), max(vt_values), 1000)
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        predictions = (vt_values >= thresh).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Sort by FPR to ensure monotonicity for AUC calculation
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    sorted_indices = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_indices]
    tpr_sorted = tpr_array[sorted_indices]
    thresholds_sorted = np.array(thresholds)[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    # Plot ROC curve
    plt.plot(fpr_sorted, tpr_sorted, label=f'{title} (AUC = {auc:.2f})')
    return thresholds_sorted, tpr_sorted, fpr_sorted

def create_histogram_and_roc():
    """Create histograms and ROC curves to determine optimal thresholds."""
    vt_values_raw_noise = []
    vt_values_denoised_noise = []
    vt_values_raw = []
    vt_values_denoised = []
    labels_raw = []
    labels_denoised = []
    num_histogram_simulations = 50  # Increased for better statistics
    
    # Collect VT values for histogram (noise region only)
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_simulation_for_histogram, range(num_histogram_simulations)), total=num_histogram_simulations, desc="Histogram Simulations"))
    
    for vt_raw, vt_denoised in results:
        vt_values_raw_noise.extend(vt_raw)
        vt_values_denoised_noise.extend(vt_denoised)
    
    # Collect VT values for ROC (full signal at 6 dB)
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_simulation_for_roc, range(num_histogram_simulations)), total=num_histogram_simulations, desc="ROC Simulations"))
    
    for vt_raw, vt_denoised, labels in results:
        vt_values_raw.extend(vt_raw)
        vt_values_denoised.extend(vt_denoised)
        labels_raw.extend(labels)
        labels_denoised.extend(labels)
    
    # Plot histograms with thresholds using noise region data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(vt_values_raw_noise, bins=50, density=True, alpha=0.5, color='blue', label='Raw VT Values')
    ax.hist(vt_values_denoised_noise, bins=50, density=True, alpha=0.5, color='green', label='Denoised VT Values')
    ax.set_title('Histogram of Variance Trajectory Values (Noise Region)')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Density')
    ax.grid(True)
    
    # Set thresholds based on 99th percentile of noise region
    threshold_raw = np.percentile(vt_values_raw_noise, 99)
    threshold_denoised = np.percentile(vt_values_denoised_noise, 99)
    
    # Add thresholds to histogram
    ax.axvline(threshold_raw, color='r', linestyle='--', label=f'Raw VT Threshold ({threshold_raw:.4f})')
    ax.axvline(threshold_denoised, color='m', linestyle='--', label=f'Denoised VT Threshold ({threshold_denoised:.4f})')
    ax.legend()
    fig.savefig(os.path.join(plots_dir, 'vt_histogram.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    # Generate ROC curves using full signal data
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    thresholds_raw, tpr_raw, fpr_raw = generate_roc_curve(np.array(vt_values_raw), np.array(labels_raw), 'Raw VT')
    thresholds_denoised, tpr_denoised, fpr_denoised = generate_roc_curve(np.array(vt_values_denoised), np.array(labels_denoised), 'Denoised VT')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR)')
    ax_roc.set_title('ROC Curve at 6 dB SNR')
    ax_roc.legend()
    ax_roc.grid(True)
    fig_roc.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_roc)
    
    print(f"Selected Raw VT threshold (99th percentile of noise region): {threshold_raw:.4f}")
    print(f"Selected Denoised VT threshold (99th percentile of noise region): {threshold_denoised:.4f}")
    return threshold_raw, threshold_denoised

def run_simulation_wrapper(args):
    sim, threshold_raw, threshold_denoised = args
    return run_simulation(sim, threshold_raw, threshold_denoised)

def main():
    print("Generating histogram and ROC curves to determine optimal thresholds...")
    threshold_raw, threshold_denoised = create_histogram_and_roc()
    
    print(f"Running {num_simulations} simulations in parallel using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        simulation_results = list(tqdm(pool.imap(run_simulation_wrapper, [(sim, threshold_raw, threshold_denoised) for sim in range(num_simulations)]), total=num_simulations, desc="Simulations"))
    
    # Aggregate results across simulations
    aggregated_results = {
        'mean_errors_raw': [],
        'mean_errors_denoised': [],
        'mean_success_raw': [],
        'mean_success_denoised': []
    }
    for snr_idx in range(len(snr_db_range)):
        errors_raw_sim = [simulation_results[sim]['errors_raw'][snr_idx] for sim in range(num_simulations)]
        errors_denoised_sim = [simulation_results[sim]['errors_denoised'][snr_idx] for sim in range(num_simulations)]
        success_raw_sim = [simulation_results[sim]['success_raw'][snr_idx] for sim in range(num_simulations)]
        success_denoised_sim = [simulation_results[sim]['success_denoised'][snr_idx] for sim in range(num_simulations)]
        
        finite_errors_raw = [abs(e) if e != float('inf') else 1000 for e in errors_raw_sim]
        finite_errors_denoised = [abs(e) if e != float('inf') else 1000 for e in errors_denoised_sim]
        mean_error_raw = np.mean(finite_errors_raw)
        mean_error_denoised = np.mean(finite_errors_denoised)
        mean_success_raw = 100 * sum(success_raw_sim) / len(success_raw_sim)
        mean_success_denoised = 100 * sum(success_denoised_sim) / len(success_denoised_sim)
        
        aggregated_results['mean_errors_raw'].append(mean_error_raw)
        aggregated_results['mean_errors_denoised'].append(mean_error_denoised)
        aggregated_results['mean_success_raw'].append(mean_success_raw)
        aggregated_results['mean_success_denoised'].append(mean_success_denoised)
    
    # Compute low-SNR metrics
    low_snr_indices = [i for i, snr in enumerate(snr_db_range) if snr < 12]
    low_snr_errors_raw = [aggregated_results['mean_errors_raw'][i] for i in low_snr_indices]
    low_snr_errors_denoised = [aggregated_results['mean_errors_denoised'][i] for i in low_snr_indices]
    low_snr_success_raw = [aggregated_results['mean_success_raw'][i] for i in low_snr_indices]
    low_snr_success_denoised = [aggregated_results['mean_success_denoised'][i] for i in low_snr_indices]
    
    mean_error_raw_low = np.mean([e for e in low_snr_errors_raw if e != float('inf')])
    mean_error_denoised_low = np.mean([e for e in low_snr_errors_denoised if e != float('inf')])
    success_rate_raw_low = np.mean(low_snr_success_raw)
    success_rate_denoised_low = np.mean(low_snr_success_denoised)
    
    print("\nAggregated Performance Metrics (Window Size 320):")
    print(f"  Overall Mean Raw VT Error: {np.mean([e for e in aggregated_results['mean_errors_raw'] if e != float('inf')])} samples")
    print(f"  Overall Mean Denoised VT Error: {np.mean([e for e in aggregated_results['mean_errors_denoised'] if e != float('inf')])} samples")
    print(f"  Overall Raw VT Success Rate: {np.mean(aggregated_results['mean_success_raw']):.2f}%")
    print(f"  Overall Denoised VT Success Rate: {np.mean(aggregated_results['mean_success_denoised']):.2f}%")
    print(f"\nLow-SNR Metrics (SNR < 12 dB):")
    print(f"  Mean Raw VT Error: {mean_error_raw_low} samples")
    print(f"  Mean Denoised VT Error: {mean_error_denoised_low} samples")
    print(f"  Raw VT Success Rate: {success_rate_raw_low:.2f}%")
    print(f"  Denoised VT Success Rate: {success_rate_denoised_low:.2f}%")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Trigger Error vs SNR
    ax1.plot(snr_db_range, aggregated_results['mean_errors_raw'], 'b-o', label='Raw VT Error')
    ax1.plot(snr_db_range, aggregated_results['mean_errors_denoised'], 'g-^', label='Denoised VT Error')
    ax1.axhline(int(0.5e-6 * fs), color='r', linestyle='--', label='Acceptable Error (±0.5 µs)')
    ax1.axhline(-int(0.5e-6 * fs), color='r', linestyle='--')
    ax1.axvline(12, color='gray', linestyle='--', label='SNR 12 dB')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Trigger Error (samples)')
    ax1.set_title('Trigger Error vs SNR (Method: VT with Histogram Threshold, Window Size 320)')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Detection Success Rate vs SNR
    ax2.plot(snr_db_range, aggregated_results['mean_success_raw'], 'b-o', label='Raw VT Success Rate')
    ax2.plot(snr_db_range, aggregated_results['mean_success_denoised'], 'g-^', label='Denoised VT Success Rate')
    ax2.axvline(12, color='gray', linestyle='--', label='SNR 12 dB')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Detection Success Rate (%)')
    ax2.set_title('Detection Success Rate vs SNR (Method: VT with Histogram Threshold, Window Size 320)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'histogram_threshold_vt_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Plot saved as '{os.path.join(plots_dir, 'histogram_threshold_vt_comparison.png')}'")
    print(f"VT time plot saved as '{os.path.join(plots_dir, 'vt_time_plot.png')}'")
    print(f"ROC curve saved as '{os.path.join(plots_dir, 'roc_curve.png')}'")

if __name__ == "__main__":
    main()
# %%
