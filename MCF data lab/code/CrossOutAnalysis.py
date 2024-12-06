import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

def load_data_from_file():
    """
    Opens a file dialog to select a data file and loads it as a NumPy array.

    Returns:
        np.array: Loaded data as a NumPy array.
    """
    default_dir = Path.cwd()
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = askopenfilename(
        title="Select a data file",
        initialdir=default_dir,
        filetypes=[("Data Files", "*.dat"), ("All Files", "*.*")]
    )

    if not file_path or not Path(file_path).exists():
        print("No valid file selected. Exiting...")
        return None

    try:
        data = np.loadtxt(file_path)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def gaussian(x, a, sigma,y_0):
    """
    Gaussian function for curve fitting.
    """
    return y_0 + a * np.exp(-((x - 6943) ** 2) / (2 * sigma ** 2)) # 6943 due to laser light

def fit_gaussian(lambda_row, intensity_row):
    """
    Fits a Gaussian curve to the data and returns the parameters and uncertainties.

    Args:
        lambda_row (np.array): Wavelength data.
        intensity_row (np.array): Intensity data.

    Returns:
        tuple: Fitted parameters (a, x0, sigma) and their uncertainties.
    """
    try:
        popt, pcov = curve_fit(gaussian, lambda_row, intensity_row, 
                               p0=[max(intensity_row), 50,0])
        perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties
        return popt, perr
    except Exception as e:
        print(f"Error in Gaussian fitting: {e}")
        return None, None

def filter_data(lambda_row, intensity_row, exclusion_ranges):
    """
    Filters intensity and wavelength data based on exclusion ranges.

    Args:
        lambda_row (np.array): Wavelength data.
        intensity_row (np.array): Intensity data.
        exclusion_ranges (list of tuples): Ranges to exclude, e.g., [(6450, 6700), (6850, 7050)].

    Returns:
        tuple: Filtered wavelength and intensity data.
    """
    mask = np.ones_like(lambda_row, dtype=bool)
    for start, end in exclusion_ranges:
        mask &= ~((start <= lambda_row) & (lambda_row <= end))
    return lambda_row[mask], intensity_row[mask]

def analyze_row(intensity_data, lambda_data, pixel_y, exclusion_ranges, show_graph=False):
    """
    Analyzes a specific row of data, fits a Gaussian, and optionally plots the results.

    Args:
        intensity_data (np.array): Intensity data array.
        lambda_data (np.array): Wavelength data array.
        pixel_y (int): Row index to analyze.
        exclusion_ranges (list of tuples): Ranges to exclude from fitting.
        show_graph (bool): Whether to display a graph.

    Returns:
        dict: Gaussian fit parameters and uncertainties.
    """
    if pixel_y < 0 or pixel_y >= intensity_data.shape[0]:
        print(f"Error: pixel_y={pixel_y} is out of bounds for the data.")
        return None

    intensity_row = intensity_data[pixel_y, :]
    lambda_row = lambda_data[pixel_y, :]
    filtered_lambda, filtered_intensity = filter_data(lambda_row, intensity_row, exclusion_ranges)
    fit_params, fit_errors = fit_gaussian(filtered_lambda, filtered_intensity)

    if show_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_row, intensity_row, label=f"Pixel Y={pixel_y}", alpha=0.6)
        plt.axvline(6943, color='purple', linestyle='--', linewidth=1.5, label="Laser Light (6943 Å)")
        for start, end in exclusion_ranges:
            plt.axvline(start, color='gray', linestyle='--', linewidth=1)
            plt.axvline(end, color='gray', linestyle='--', linewidth=1)
        if fit_params is not None:
            fit_x = np.linspace(min(filtered_lambda), max(filtered_lambda), 500)
            fit_y = gaussian(fit_x, *fit_params)
            plt.plot(fit_x, fit_y, 'r-', label="Fitted Gaussian", linewidth=2)
        plt.xlabel("Real Wavelength Å")
        plt.xlim(5800,9000)
        plt.ylabel("Intensity")
        plt.title(f"Intensity vs. Real Wavelength at y={pixel_y}")
        plt.legend()
        plt.grid(True)
        plt.show()

    if fit_params is not None:
        return {
            "amplitude": (fit_params[0], fit_errors[0]),
            "sigma": (fit_params[1], fit_errors[1]),
            "noise": (fit_params[2], fit_errors[2])
        }
    return None

def analyze_multiple_rows(intensity_data, lambda_data, pixel_y_values, exclusion_ranges):
    """
    Analyzes multiple rows and returns the Gaussian fit parameters for each.

    Args:
        intensity_data (np.array): Intensity data array.
        lambda_data (np.array): Wavelength data array.
        pixel_y_values (list of int): Row indices to analyze.
        exclusion_ranges (list of tuples): Ranges to exclude from fitting.

    Returns:
        dict: Gaussian fit results for each row index.
    """
    results = {}
    for pixel_y in pixel_y_values:
        results[pixel_y] = analyze_row(intensity_data, lambda_data, pixel_y, exclusion_ranges, show_graph=False)
    return results

if __name__ == "__main__":
    print("Select the intensity data file:")
    intensity_data = load_data_from_file()

    print("Select the lambda mapping file:")
    lambda_data = load_data_from_file()

    if intensity_data is not None and lambda_data is not None:
        if intensity_data.shape != lambda_data.shape:
            print("Error: Intensity data and lambda data must have the same shape.")
        else:
            exclusion_ranges = [(0, 5800), (9000, 20000),(6450, 6700), (6850, 7050)]
            
            # Example analysis with a graph
            print("Analyzing pixel_y=15 with graph:")
            results = analyze_row(intensity_data, lambda_data, 15, exclusion_ranges, show_graph=True)
            print(results)

            # Automated analysis for multiple rows
            #print("Analyzing multiple rows:")
            #rows_to_analyze = [100, 150, 200]
            #results = analyze_multiple_rows(intensity_data, lambda_data, rows_to_analyze, exclusion_ranges)
            #print("Results:", results)
    else:
        print("One or both data arrays could not be loaded. Exiting...")