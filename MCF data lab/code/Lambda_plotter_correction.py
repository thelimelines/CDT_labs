import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

if __name__ == "__main__":
    print("Select the intensity data file:")
    intensity_data = load_data_from_file()

    print("Select the lambda mapping file:")
    lambda_data = load_data_from_file()

    if intensity_data is not None and lambda_data is not None:
        # Ensure both arrays have the same shape
        if intensity_data.shape != lambda_data.shape:
            print("Error: Intensity data and lambda data must have the same shape.")
        else:
            y_pixels, x_pixels = intensity_data.shape

            # Determine the common wavelength grid
            min_wavelength = np.min(lambda_data)
            max_wavelength = np.max(lambda_data)
            common_wavelength = np.linspace(min_wavelength, max_wavelength, 1000)

            # Initialize the new interpolated intensity map
            interpolated_intensity = np.zeros((y_pixels, len(common_wavelength)))

            for y in range(y_pixels):
                # Extract the corresponding intensity and wavelength rows
                intensity_row = intensity_data[y, :]
                lambda_row = lambda_data[y, :]

                # Interpolate the intensity data to the common wavelength grid
                interp_func = interp1d(lambda_row, intensity_row, kind='linear', bounds_error=False, fill_value=0)
                interpolated_intensity[y, :] = interp_func(common_wavelength)

            # Create the heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(
                interpolated_intensity,
                aspect='auto',
                extent=[common_wavelength[0], common_wavelength[-1], 0, y_pixels],
                origin='lower',
                cmap='plasma'  # Change colormap as desired
            )
            plt.colorbar(label="Intensity")
            plt.xlabel("Real Wavelength")
            plt.ylabel("Pixel Y")
            plt.title("Intensity Heatmap with Real Wavelength Scaling")
            plt.grid(False)  # Turn off the grid for better visualization
            plt.xlim(5500,9000)
            plt.show()
    else:
        print("One or both data arrays could not be loaded. Exiting...")