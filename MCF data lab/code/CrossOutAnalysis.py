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
            # Define the pixel_y index
            pixel_y = 150

            # Ensure the pixel_y index is within bounds
            if pixel_y < 0 or pixel_y >= intensity_data.shape[0]:
                print(f"Error: pixel_y={pixel_y} is out of bounds for the data.")
            else:
                # Extract the row for the given pixel_y
                intensity_row = intensity_data[pixel_y, :]
                lambda_row = lambda_data[pixel_y, :]

                # Plot intensity vs. real wavelength
                plt.figure(figsize=(10, 6))
                plt.plot(lambda_row, intensity_row, label=f"Pixel Y={pixel_y}")
                plt.xlabel("Real Wavelength")
                plt.ylabel("Intensity")
                plt.title("Intensity vs. Real Wavelength at y=150")
                plt.legend()
                plt.grid(True)
                plt.show()
    else:
        print("One or both data arrays could not be loaded. Exiting...")
