# %%
# reimplement_denoised_vt.py
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, analyze_noise
from denoising import denoise_signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Define parameters
fs = 20e6  # Sampling frequency in Hz
base_seed = 42  # Base seed for reproducibility
num_simulations = 50  # Maximum smoothing as in dissertation
debug_level = 0  # Disable debug prints
window_size = 320  # Fixed at 320 samples (16 µs)
snr_db_range = np.concatenate([np.arange(-3, 12, 1), np.arange(12, 31, 6)])  # -3 to 11 dB (1 dB), 12 to 30 dB (6 dB)
snr_low_db_range = np.arange(-3, 12, 1)  # Low SNR range for metric (-3 to 11 dB)
noise_duration = 80e-6  # 80 µs to include preamble transition
noise_samples = int(noise_duration * fs)
threshold_multiplier = 0.5  # Denoising threshold multiplier, per dissertation
vt_threshold_multiplier = 3.0  # VT threshold multiplier, to be calibrated

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

def evaluate_threshold_method(input_signal, t, snr_db):
    """Evaluate using both Raw VT and Denoised VT with window_size = 320."""
    effective_window_size = min(window_size, len(input_signal) - 1)
    
    # Raw VT analysis
    noise_stats_raw = analyze_noise(input_signal, t, window_size=effective_window_size, 
                                   noise_region=(0, noise_duration), debug_level=debug_level)
    threshold_raw = noise_stats_raw['mean_vt'] + vt_threshold_multiplier * noise_stats_raw['std_vt']
    variance_traj_raw, _, _, trigger_idx_raw = variance_trajectory_detector(
        input_signal, t, window_size=effective_window_size, threshold=threshold_raw, debug_level=debug_level
    )
    trigger_idx_raw_full = trigger_idx_raw + effective_window_size - 1 if trigger_idx_raw >= 0 else -1
    error_raw = trigger_idx_raw_full - noise_samples if trigger_idx_raw_full >= 0 else float('inf')
    success_raw = abs(error_raw) <= int(1e-6 * fs) if error_raw != float('inf') else False
    
    # Denoised VT analysis
    denoised_signal = denoise_signal(input_signal, t, threshold_multiplier=threshold_multiplier, debug_level=debug_level)
    noise_stats_denoised = analyze_noise(denoised_signal, t, window_size=effective_window_size, 
                                        noise_region=(0, noise_duration), debug_level=debug_level)
    threshold_denoised = noise_stats_denoised['mean_vt'] + vt_threshold_multiplier * noise_stats_denoised['std_vt']
    variance_traj_denoised, _, _, trigger_idx_denoised = variance_trajectory_detector(
        denoised_signal, t, window_size=effective_window_size, threshold=threshold_denoised, debug_level=debug_level
    )
    trigger_idx_denoised_full = trigger_idx_denoised + effective_window_size - 1 if trigger_idx_denoised >= 0 else -1
    error_denoised = trigger_idx_denoised_full - noise_samples if trigger_idx_denoised_full >= 0 else float('inf')
    success_denoised = abs(error_denoised) <= int(1e-6 * fs) if error_denoised != float('inf') else False
    
    return error_raw, error_denoised, success_raw, success_denoised

def run_simulation(sim):
    """Run a single Monte Carlo simulation comparing Raw VT and Denoised VT."""
    np.random.seed(base_seed + sim)
    
    t_clean, preamble_clean = generate_80211ag_preamble(fs=fs, add_rf_fingerprint=True, seed=base_seed + sim, debug_level=debug_level)
    
    sim_results = {'errors_raw': [], 'errors_denoised': [], 'success_raw': [], 'success_denoised': []}
    
    for snr_db in snr_db_range:
        noisy_signal, noise_std = add_noise_to_signal(preamble_clean, snr_db, fs)
        noise = (np.random.randn(noise_samples) + 1j * np.random.randn(noise_samples)) * noise_std / np.sqrt(2)
        t_noise = np.arange(noise_samples) / fs
        input_signal = np.concatenate([noise, noisy_signal])
        t = np.concatenate([t_noise, t_noise[-1] + (1 / fs) + t_clean])
        
        error_raw, error_denoised, success_raw, success_denoised = evaluate_threshold_method(
            input_signal, t, snr_db
        )
        sim_results['errors_raw'].append(error_raw)
        sim_results['errors_denoised'].append(error_denoised)
        sim_results['success_raw'].append(success_raw)
        sim_results['success_denoised'].append(success_denoised)
    
    return sim_results

def main():
    print(f"Running {num_simulations} simulations in parallel using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        simulation_results = []
        for result in tqdm(pool.imap(run_simulation, range(num_simulations)), total=num_simulations, desc="Simulations"):
            simulation_results.append(result)
    
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
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Trigger Error vs SNR
    plt.subplot(2, 1, 1)
    plt.plot(snr_db_range, aggregated_results['mean_errors_raw'], 'b-o', label='Raw VT Error')
    plt.plot(snr_db_range, aggregated_results['mean_errors_denoised'], 'g-^', label='Denoised VT Error')
    plt.axhline(int(1e-6 * fs), color='r', linestyle='--', label='Acceptable Error (±1 µs)')
    plt.axhline(-int(1e-6 * fs), color='r', linestyle='--')
    plt.axvline(12, color='gray', linestyle='--', label='SNR 12 dB')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Trigger Error (samples)')
    plt.title('Trigger Error vs SNR (Method: VT, Window Size 320)')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Detection Success Rate vs SNR
    plt.subplot(2, 1, 2)
    plt.plot(snr_db_range, aggregated_results['mean_success_raw'], 'b-o', label='Raw VT Success Rate')
    plt.plot(snr_db_range, aggregated_results['mean_success_denoised'], 'g-^', label='Denoised VT Success Rate')
    plt.axvline(12, color='gray', linestyle='--', label='SNR 12 dB')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Success Rate (%)')
    plt.title('Detection Success Rate vs SNR (Method: VT, Window Size 320)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reimplement_denoised_vt_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'reimplement_denoised_vt_comparison.png' in the current directory")

if __name__ == "__main__":
    main()
# %%
