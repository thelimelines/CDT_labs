import numpy as np
import matplotlib.pyplot as plt
from CrossOutAnalysis import load_data_from_file, analyze_multiple_rows
from scipy.constants import c, m_e, k

# Constants
lambda_0 = 6943e-10  # Central wavelength in meters (convert from angstroms)
c = c  # Speed of light in m/s
m_e = m_e  # Electron mass in kg
k_B = k  # Boltzmann constant in J/K

def calculate_temperature(sigma, theta):
    """
    Calculates the temperature (T_e) based on the given equation.
    
    Args:
        sigma (float): Gaussian sigma in meters (converted from angstroms).
        theta (float): Scattering angle in radians.

    Returns:
        float: Calculated temperature in kelvins.
    """
    sigma_m = sigma * 1e-10  # Convert sigma from angstroms to meters
    temperature = (m_e / k_B) * ((sigma_m * c) / (2 * np.sin(theta / 2) * lambda_0))**2
    return temperature

# Load data files
intensity_data = load_data_from_file()
lambda_data = load_data_from_file()
angle_data = load_data_from_file()  # Assuming angles are in degrees

if intensity_data is not None and lambda_data is not None:
    if intensity_data.shape != lambda_data.shape:
        print("Error: Intensity data and lambda data must have the same shape.")
    else:
        exclusion_ranges = [(0, 5800), (9000, 20000),(6450, 6700), (6850, 7050)]
        y_values = range(intensity_data.shape[0])  # Loop through all y values
        
        # Analyze rows and get Gaussian fit results
        fit_results = analyze_multiple_rows(intensity_data, lambda_data, y_values, exclusion_ranges)
        
        # Prepare data for plotting
        temperatures = []
        temp_uncertainties = []
        angles = angle_data[:len(fit_results)]  # Match angles with rows
        
        for y, fit in fit_results.items():
            if fit is not None and y < len(angle_data):
                sigma, sigma_err = fit["sigma"]
                
                # Skip if the fit is invalid (e.g., sigma_err is NaN or negative)
                if np.isnan(sigma_err) or sigma_err <= 0:
                    print(f"Skipping y={y}: Invalid sigma_err={sigma_err}")
                    continue
                
                theta = angles[y]
                temperature = calculate_temperature(sigma, theta)
                temperature_err = temperature * (sigma_err / sigma)  # Propagate uncertainty
                
                # Skip if uncertainty is invalid
                if temperature_err < 0 or np.isnan(temperature_err):
                    print(f"Skipping y={y}: Invalid temperature_err={temperature_err}")
                    continue
                
                temperatures.append(temperature)
                temp_uncertainties.append(temperature_err)
        
        # Plot results
        if len(temperatures) > 0:
            plt.figure(figsize=(10, 6))
            plt.errorbar(angles[:len(temperatures)], temperatures, yerr=temp_uncertainties, fmt='o', label="Temperature vs Angle")
            plt.xlabel("Scattering Angle (radians)")
            #plt.xlim(1.4,2)
            plt.ylabel("Temperature (K)")
            plt.ylim(-1e6,1.5e7)
            plt.title("Temperature vs Scattering Angle")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            print("No valid data to plot.")

else:
    print("Failed to load one or both data arrays.")