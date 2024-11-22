import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

def load_data_from_file():
    """
    Opens a file dialog to select a data file and loads it as a 2D NumPy array.

    Returns:
        np.array: Loaded data as a 2D NumPy array.
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

def plot_intensity_image(data):
    """
    Plots a 2D array as an image.
    
    Args:
        data (np.array): 2D array to visualize.
    """
    if data is None:
        print("No data available for plotting.")
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title("CCD Detector Intensity Data")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()

if __name__ == "__main__":
    data = load_data_from_file()
    plot_intensity_image(data)