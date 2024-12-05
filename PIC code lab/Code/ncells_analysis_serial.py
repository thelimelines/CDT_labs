import numpy as np
import matplotlib.pyplot as plt
from Electrostatic_PIC_1D import landau, run, Summary
from Harmonic_peak_plotter import analyze_harmonic_data
from numpy import linspace, pi
import time

# Simulation parameters
npart = 10000
L = 4 * pi  # Domain length
random_seed = 10
output_times = linspace(0, 20, 50)  # 50 output points between t=0 and t=20

# Initialize arrays for storing results
ncells_range = range(10, 210, 10)  # Iterating over ncells from 10 to 50 in steps of 5
simulation_times = []
rms_noise_values = []
omega_values = []
omega_uncertainties = []
damping_rates = []
damping_uncertainties = []

# Run the simulation for each ncells value
for ncells in ncells_range:
    # Generate initial conditions
    np.random.seed(random_seed)
    pos, vel = landau(npart, L)

    # Create a Summary object for diagnostics
    summary = Summary()

    # Measure simulation time
    start_time = time.time()
    run(pos, vel, L, ncells=ncells, out=[summary], output_times=output_times)
    end_time = time.time()
    simulation_times.append(end_time - start_time)

    # Analyze the collected harmonic data
    times = np.array(summary.t)
    harmonics = np.array(summary.firstharmonic)
    A0, d, A0_uncertainty, d_uncertainty, omega, omega_uncertainty, noise_rms, _ = analyze_harmonic_data(
        times, harmonics, plot=False
    )

    # Store results
    rms_noise_values.append(noise_rms)
    omega_values.append(omega)
    omega_uncertainties.append(omega_uncertainty)
    damping_rates.append(d)
    damping_uncertainties.append(d_uncertainty)

# Plot results in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Simulation Time vs ncells
axes[0, 0].plot(ncells_range, simulation_times, marker='o', label="Simulation Time")
axes[0, 0].set_title("Simulation Time vs ncells")
axes[0, 0].set_xlabel("Number of Cells (ncells)")
axes[0, 0].set_ylabel("Time (s)")
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: RMS Noise vs ncells
axes[0, 1].plot(ncells_range, rms_noise_values, marker='o', label="RMS Noise")
axes[0, 1].set_title("RMS Noise vs ncells")
axes[0, 1].set_xlabel("Number of Cells (ncells)")
axes[0, 1].set_ylabel("RMS Noise")
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot 3: Harmonic Frequency vs ncells
axes[1, 0].errorbar(ncells_range, omega_values, yerr=omega_uncertainties, fmt='o', label="Harmonic Frequency (Omega)")
axes[1, 0].set_title("Harmonic Frequency vs ncells")
axes[1, 0].set_xlabel("Number of Cells (ncells)")
axes[1, 0].set_ylabel("Frequency (rad/s)")
axes[1, 0].grid(True)
axes[1, 0].legend()

# Plot 4: Damping Rate vs ncells
axes[1, 1].errorbar(ncells_range, damping_rates, yerr=damping_uncertainties, fmt='o', label="Damping Rate (Gamma)")
axes[1, 1].set_title("Damping Rate vs ncells")
axes[1, 1].set_xlabel("Number of Cells (ncells)")
axes[1, 1].set_ylabel("Damping Rate")
axes[1, 1].grid(True)
axes[1, 1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()