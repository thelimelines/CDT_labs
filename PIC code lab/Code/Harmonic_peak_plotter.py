import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import hilbert

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


def analyze_harmonic_data(times, harmonics):
    """
    Analyzes the first harmonic data for peaks, noise, and signal-to-noise separation.

    Args:
        times (np.array): Array of time points.
        harmonics (np.array): Array of harmonic amplitudes.

    Returns:
        peaks_times (np.array): Times of detected peaks.
        peaks_amplitudes (np.array): Amplitudes of detected peaks.
        noise_level (float): Calculated noise level amplitude.
    """
    # --- Find Peaks ---
    # Use scipy's find_peaks to detect peaks in the data
    peaks, _ = find_peaks(harmonics, height=0)
    peaks_times = times[peaks]
    peaks_amplitudes = harmonics[peaks]

    # --- Identify Noise Start ---
    # Find the first peak where the amplitude is greater than the previous one
    for i in range(1, len(peaks_amplitudes)):
        if peaks_amplitudes[i] > peaks_amplitudes[i - 1]:
            noise_start_index = peaks[i]
            break
    else:
        noise_start_index = len(harmonics)  # If no such peak exists, use the full range

    # Split data into signal and noise
    signal = harmonics[:noise_start_index]
    noise = harmonics[noise_start_index:]

    # --- Calculate Noise Level ---
    # Define noise level as the standard deviation of the noise data
    noise_level = np.std(noise)

    # --- Plot Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, harmonics, label="Harmonic Amplitude")
    plt.scatter(peaks_times, peaks_amplitudes, color='red', marker='x', label="Detected Peaks")
    plt.axvline(times[noise_start_index], color='green', linestyle='--', label="Noise Start")
    plt.axhline(noise_level, color='purple', linestyle='--', label="Min Noise Level")
    plt.xlabel("Time [Normalised]")
    plt.ylabel("Harmonic Amplitude [Normalised]")
    plt.yscale("log")
    plt.title("Harmonic Amplitude Analysis")
    plt.legend()
    plt.grid(True)

    plt.ioff() # This so that the windows stay open
    plt.show()

    return peaks_times, peaks_amplitudes, noise_level

if __name__ == "__main__":
    # Load the data from a file
    data = load_data_from_file()
    if data is not None:
        times, harmonics = data[:, 0], data[:, 1]

        # Analyze harmonic data
        peaks_times, peaks_amplitudes, noise_level = analyze_harmonic_data(times, harmonics)
