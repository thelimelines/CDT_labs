import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

def load_and_plot_data():
    """Allows the user to select a CSV file, reads the harmonic data, and plots it."""
    
    # Set the default directory to the current directory
    default_dir = Path.cwd()

    # Create a Tkinter root window but hide it
    root = Tk()
    root.withdraw()  # Hides the small root window
    root.attributes('-topmost', True)  # Brings the dialog to the front

    # Open a file dialog to select the file
    file_path = askopenfilename(
        title="Select a harmonic data file",
        initialdir=default_dir,  # Default to the current directory
        filetypes=[("CSV Files", "*.csv")]
    )
    
    if not file_path or not Path(file_path).exists():
        print("No valid file selected. Exiting...")
        return

    # Read the data from the selected file
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip header row
        times = data[:, 0]  # First column: time
        harmonics = data[:, 1]  # Second column: first harmonic amplitude
        
        # Plot the data
        plt.figure()
        plt.plot(times, harmonics, label="First Harmonic Amplitude")
        plt.xlabel("Time [Normalised]")
        plt.ylabel("First Harmonic Amplitude [Normalised]")
        plt.title("Harmonic Amplitude vs Time")
        plt.yscale("log")

        plt.ioff() # This so that the windows stay open
        plt.show()
        
    except Exception as e:
        print(f"Error reading or plotting file: {e}")

if __name__ == "__main__":
    load_and_plot_data()
