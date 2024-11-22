import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

def load_data_from_file():
    """
    Opens a file dialog to select a data file and loads it as a NumPy array.

    Returns:
        np.array: Loaded data as a NumPy array.
    """
    # Set the default directory to the current working directory
    default_dir = Path.cwd()

    # Create a Tkinter root window but hide it
    root = Tk()
    root.withdraw()  # Hides the small root window
    root.attributes('-topmost', True)  # Brings the dialog to the front

    # Open a file dialog to select the file
    file_path = askopenfilename(
        title="Select a data file",
        initialdir=default_dir,
        filetypes=[("Data Files", "*.dat"), ("All Files", "*.*")]
    )

    # Check if a valid file was selected
    if not file_path or not Path(file_path).exists():
        print("No valid file selected. Exiting...")
        return None

    # Load the data from the file
    try:
        data = np.loadtxt(file_path)  # Default whitespace delimiter
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def map_wavelengths(intensity_data, lambda_data):
    """
    Maps the intensity data to wavelengths using lambda data.

    Args:
        intensity_data (np.array): The intensity data (2D array).
        lambda_data (np.array): The lambda data (2D array, same shape as intensity_data).

    Returns:
        np.array, np.array, np.array: Intensity data, wavelength (x-axis), and space (y-axis).
    """
    if intensity_data is None or lambda_data is None:
        print("Intensity data or lambda data is missing.")
        return None, None, None

    # Validate that the dimensions match
    if intensity_data.shape != lambda_data.shape:
        print("Mismatch between intensity data and lambda data dimensions.")
        return None, None, None

    # Extract the wavelength axis (unique values along columns for the x-axis)
    wavelengths = np.unique(lambda_data[0, :])  # First row represents x-axis wavelengths
    space = np.arange(intensity_data.shape[0])  # Y-axis corresponds to spatial indices

    return intensity_data, wavelengths, space

def plot_corrected_image(intensity_data, lambda_data):
    """
    Plots the corrected intensity data using wavelengths for x-axis and space for y-axis.

    Args:
        intensity_data (np.array): The intensity data (2D array).
        lambda_data (np.array): The lambda mapping (2D array, same shape as intensity_data).
    """
    if intensity_data is None or lambda_data is None:
        print("Insufficient data for plotting.")
        return

    # Plot the corrected data
    plt.figure(figsize=(12, 8))
    plt.imshow(intensity_data, extent=[
               lambda_data.min(), lambda_data.max(), 0, intensity_data.shape[0]],
               cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title("Corrected Intensity Data")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Space (pixels)")
    plt.show()

if __name__ == "__main__":
    print("Select the intensity data file:")
    intensity_data = load_data_from_file()

    print("Select the lambda mapping file:")
    lambda_data = load_data_from_file()

    if intensity_data is not None and lambda_data is not None:
        # Ensure dimensions match
        if intensity_data.shape != lambda_data.shape:
            print("Error: Intensity data and lambda data must have the same dimensions.")
        else:
            # Flatten both arrays for reordering
            wavelengths = lambda_data.flatten()
            intensities = intensity_data.flatten()

            # Sort by wavelength for proper x-axis ordering
            sorted_indices = np.argsort(wavelengths)
            sorted_wavelengths = wavelengths[sorted_indices]
            sorted_intensities = intensities[sorted_indices]

            # Create a 2D histogram for plotting (wavelength on x, spatial index on y)
            unique_wavelengths = np.unique(sorted_wavelengths)
            space = np.arange(intensity_data.shape[0])

            corrected_image = np.zeros((len(space), len(unique_wavelengths)))

            for y in range(intensity_data.shape[0]):
                for x in range(intensity_data.shape[1]):
                    wavelength = lambda_data[y, x]
                    wavelength_index = np.where(unique_wavelengths == wavelength)[0][0]
                    corrected_image[y, wavelength_index] += intensity_data[y, x]

            # Plot the corrected image
            corrected_image = np.fliplr(corrected_image) # Flip data for better visualisation
            plt.figure(figsize=(12, 8))
            plt.imshow(corrected_image, extent=[
                unique_wavelengths.min(), unique_wavelengths.max(), 0, intensity_data.shape[0]],
                cmap="plasma", origin="lower", aspect="auto")
            plt.colorbar(label="Intensity")
            plt.title("Corrected Intensity Data")
            plt.xlabel("Wavelength (Angstroms)")
            plt.ylabel("Space (pixels)")
            plt.show()