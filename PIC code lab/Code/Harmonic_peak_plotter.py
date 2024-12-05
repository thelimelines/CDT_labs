import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def load_data_from_file():
    """
    Opens a file dialog to select a CSV file and loads the data into a NumPy array.

    Returns:
        np.array: Loaded data as a 2D NumPy array, where the first column is time and the second column is the harmonic amplitude.
    """
    # Set the default directory to the current working directory
    default_dir = Path.cwd()  # Current working directory

    # Create a Tkinter root window but hide it
    root = Tk()
    root.withdraw()  # Hides the small root window
    root.attributes('-topmost', True)  # Brings the dialog to the front

    # Open a file dialog to select the file
    file_path = askopenfilename(
        title="Select a harmonic data file",
        initialdir=default_dir,
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    # Check if a valid file was selected
    if not file_path or not Path(file_path).exists():
        print("No valid file selected. Exiting...")
        return None

    # Load the data from the CSV file
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip the header
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def exponential_model(t, A0, d):
    """Exponential decay model."""
    return A0 * np.exp(-d * t)

def fit_exponential_to_peaks(times, peaks_amplitudes):
    """
    Fits an exponential decay to the detected peaks.
    
    Args:
        times (np.array): Times of detected peaks.
        peaks_amplitudes (np.array): Amplitudes of detected peaks.
    
    Returns:
        popt (tuple): Optimal parameters (A0, d).
        pcov (np.array): Covariance matrix of the fit.
    """
    # Perform curve fitting
    popt, pcov = curve_fit(exponential_model, times, peaks_amplitudes, p0=(peaks_amplitudes[0], 0.1))
    return popt, pcov

def calculate_frequency_and_uncertainty(times):
    """
    Calculates the angular frequency (omega) and its uncertainty based on the time intervals between peaks.

    Args:
        times (np.array): Times of detected peaks.

    Returns:
        omega (float): Angular frequency (rad/s).
        omega_uncertainty (float): Uncertainty in angular frequency (rad/s).
    """
    # --- Calculate Time Intervals ---
    # Compute the time intervals between consecutive peaks
    time_intervals = np.diff(times)

    # Calculate the mean and standard deviation of the time intervals
    mean_dt = np.mean(time_intervals)
    std_dt = np.std(time_intervals)

    # --- Calculate Frequency and Uncertainty ---
    # Omega is defined as pi divided by the mean time interval
    omega = np.pi / mean_dt

    # Uncertainty in omega is derived using error propagation
    omega_uncertainty = (np.pi / mean_dt**2) * std_dt

    return omega, omega_uncertainty

def analyze_harmonic_data(times, harmonics, plot=True):
    """
    Analyzes the harmonic data for peaks, noise, and fits an exponential decay
    to the peaks above a 2x noise threshold. Also calculates the frequency (omega).

    Args:
        times (np.array): Array of time points.
        harmonics (np.array): Array of harmonic amplitudes.
        plot (bool): If True, generates visualizations. Default is True.

    Returns:
        A0 (float): Initial amplitude of the fitted exponential.
        d (float): Decay rate of the fitted exponential.
        A0_uncertainty (float): Uncertainty in the initial amplitude.
        d_uncertainty (float): Uncertainty in the decay rate.
        omega (float): Angular frequency of the peaks.
        omega_uncertainty (float): Uncertainty in the angular frequency.
        noise_rms (float): RMS noise floor.
        noise_floor_threshold (float): Threshold of 2x RMS noise floor.
    """
    # --- Find Peaks ---
    peaks, _ = find_peaks(harmonics, height=0)
    peaks_times = times[peaks]
    peaks_amplitudes = harmonics[peaks]

    # --- Calculate RMS Noise Floor ---
    noise = harmonics[peaks[-1]:] if len(peaks) > 0 else harmonics
    if len(noise) > 0:
        noise_rms = np.sqrt(np.mean(noise**2))
        noise_floor_threshold = 2 * noise_rms
    else:
        print("No clear noise-dominated region found. Using default noise floor values.")
        noise_rms, noise_floor_threshold = 0, 0

    # --- Filter Peaks Above 2x Noise Floor ---
    valid_indices = peaks_amplitudes >= noise_floor_threshold
    signal_peaks_times = peaks_times[valid_indices]
    signal_peaks_amplitudes = peaks_amplitudes[valid_indices]

    # --- Fit Exponential Decay ---
    if len(signal_peaks_times) > 1:
        popt, pcov = fit_exponential_to_peaks(signal_peaks_times, signal_peaks_amplitudes)
        A0, d = popt
        A0_uncertainty, d_uncertainty = np.sqrt(np.diag(pcov))
    else:
        print("Not enough points above the 2x noise floor threshold for fitting.")
        A0, d, A0_uncertainty, d_uncertainty = 0, 0, 0, 0

    # --- Calculate Frequency ---
    if len(signal_peaks_times) > 1:
        omega, omega_uncertainty = calculate_frequency_and_uncertainty(signal_peaks_times)
    else:
        omega, omega_uncertainty = 0, 0

    # --- Identify Region Boundaries ---
    signal_dominated_end = signal_peaks_times[-1] if len(signal_peaks_times) > 0 else 0
    noise_dominated_start = times[peaks[-1]] if len(peaks) > 0 else times[-1]

    # --- Print Results ---
    print(f"Fit Parameters: A0 = {A0:.2e} ± {A0_uncertainty:.2e}, d = {d:.2e} ± {d_uncertainty:.2e}")
    print(f"Frequency (omega): {omega:.2e} ± {omega_uncertainty:.2e} rad/s")
    print(f"RMS Noise Floor: {noise_rms:.2e}")
    print(f"Threshold (2x noise floor): {noise_floor_threshold:.2e}")
    print(f"Signal Dominated Region Ends at: {signal_dominated_end:.2f}")
    print(f"Noise Dominated Region Starts at: {noise_dominated_start:.2f}")

    # --- Plot Results (if enabled) ---
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(times, harmonics, label="Harmonic Amplitude")
        plt.scatter(peaks_times, peaks_amplitudes, color='red', marker='x', label="Detected Peaks")
        plt.axvline(signal_dominated_end, color='darkgreen', linestyle='--', label="Signal-Dominated Region End")
        plt.axvline(noise_dominated_start, color='lime', linestyle='--', label="Noise-Dominated Region Start")
        plt.axhline(noise_rms, color='violet', linestyle='--', label="RMS Noise Floor")
        plt.axhline(noise_floor_threshold, color='purple', linestyle='--', label="2x Noise Floor Threshold")

        if len(signal_peaks_times) > 1:
            fit_curve = exponential_model(signal_peaks_times, A0, d)
            fit_uncertainty_upper = fit_curve * np.exp(d_uncertainty)
            fit_uncertainty_lower = fit_curve * np.exp(-d_uncertainty)
            plt.plot(signal_peaks_times, fit_curve, 'orange', linestyle='--',
                     label=f"Fit: A0*exp(-d*t)\nA0={A0:.2e}±{A0_uncertainty:.2e}, d={d:.2e}±{d_uncertainty:.2e}")
            plt.fill_between(signal_peaks_times, fit_uncertainty_lower, fit_uncertainty_upper,
                             color='orange', alpha=0.3, label="Fit Uncertainty")

        plt.xlabel("Time [Normalized]")
        plt.ylabel("Harmonic Amplitude [Normalized]")
        plt.yscale("log")
        plt.title("Harmonic Amplitude Analysis")
        plt.legend()
        plt.grid(True)
        plt.ioff()  # Ensure the windows stay open
        plt.show()

    return A0, d, A0_uncertainty, d_uncertainty, omega, omega_uncertainty, noise_rms, noise_floor_threshold

if __name__ == "__main__":
    # Load the data from a file
    data = load_data_from_file()
    if data is not None:
        times, harmonics = data[:, 0], data[:, 1]

        A0, d, A0_uncertainty, d_uncertainty, omega, omega_uncertainty, noise_rms, noise_floor_threshold = analyze_harmonic_data(times, harmonics, plot=True)