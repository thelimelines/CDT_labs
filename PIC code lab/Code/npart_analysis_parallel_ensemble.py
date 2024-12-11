from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from Electrostatic_PIC_1D import landau, run, Summary
from Harmonic_peak_plotter import analyze_harmonic_data
from numpy import linspace, pi
import time

# Simulation parameters
ncells = 40  # Number of cells (fixed)
L = 4 * pi  # Domain length
output_times = linspace(0, 20, 50)  # 50 output points between t=0 and t=20
trials = 10  # Number of trials per npart
# Logarithmic range of npart to simulate
npart_range = np.unique(np.logspace(2, 5, num=20, dtype=int))

# Function to execute a single simulation
def run_simulation(npart, seed):
    # Generate initial conditions with the provided random seed
    np.random.seed(seed)
    pos, vel = landau(npart, L)

    # Create a Summary object for diagnostics
    summary = Summary()

    # Measure simulation time
    start_time = time.time()
    run(pos, vel, L, ncells=ncells, out=[summary], output_times=output_times)
    end_time = time.time()

    # Analyze the collected harmonic data
    times = np.array(summary.t)
    harmonics = np.array(summary.firstharmonic)
    A0, d, A0_uncertainty, d_uncertainty, omega, omega_uncertainty, noise_rms, _ = analyze_harmonic_data(
        times, harmonics, plot=False
    )

    # Return results
    return {
        "npart": npart,
        "simulation_time": end_time - start_time,
        "rms_noise": noise_rms,
        "omega": omega,
        "omega_uncertainty": omega_uncertainty,
        "damping_rate": d,
        "damping_uncertainty": d_uncertainty,
    }

# Run simulations in parallel
results = Parallel(n_jobs=-1)(
    delayed(run_simulation)(npart, seed)
    for npart in npart_range
    for seed in range(1, trials + 1)
)

# Aggregate results by npart
aggregated_results = {}
for r in results:
    npart = r["npart"]  # Extract the number of particles for this simulation
    # Initialize storage for this npart value if not already present
    if npart not in aggregated_results:
        aggregated_results[npart] = {
            "simulation_time": [],
            "rms_noise": [],
            "omega": [],
            "omega_uncertainty": [],
            "damping_rate": [],
            "damping_uncertainty": [],
        }
    # Append the respective results for this trial to the lists
    aggregated_results[npart]["simulation_time"].append(r["simulation_time"])
    aggregated_results[npart]["rms_noise"].append(r["rms_noise"])
    if not np.isnan(r["omega"]) and not np.isnan(r["omega_uncertainty"]):
        aggregated_results[npart]["omega"].append(r["omega"])
        aggregated_results[npart]["omega_uncertainty"].append(r["omega_uncertainty"])
    if not np.isnan(r["damping_rate"]) and not np.isnan(r["damping_uncertainty"]):
        aggregated_results[npart]["damping_rate"].append(r["damping_rate"])
        aggregated_results[npart]["damping_uncertainty"].append(r["damping_uncertainty"])

# Compute means and uncertainties for aggregated data
npart_values = []               # List of unique npart values
simulation_times = []           # Mean simulation times
simulation_time_std = []        # Standard deviation of simulation times
rms_noise_values = []           # Mean RMS noise values
rms_noise_std = []              # Standard deviation of RMS noise values
omega_values = []               # Mean harmonic frequencies (omega)
omega_uncertainties = []        # Combined uncertainties of omega
damping_rates = []              # Mean damping rates (gamma)
damping_uncertainties = []      # Combined uncertainties of damping rates

for npart, data in aggregated_results.items():
    npart_values.append(npart)
    simulation_times.append(np.mean(data["simulation_time"]))
    simulation_time_std.append(np.std(data["simulation_time"], ddof=1))
    rms_noise_values.append(np.mean(data["rms_noise"]))
    rms_noise_std.append(np.std(data["rms_noise"], ddof=1))
    if data["omega"]:
        omega_values.append(np.mean(data["omega"]))
        omega_uncertainties.append(np.sqrt(np.mean(np.array(data["omega_uncertainty"])**2)))
    else:
        omega_values.append(np.nan)
        omega_uncertainties.append(np.nan)
    if data["damping_rate"]:
        damping_rates.append(np.mean(data["damping_rate"]))
        damping_uncertainties.append(np.sqrt(np.mean(np.array(data["damping_uncertainty"])**2)))
    else:
        damping_rates.append(np.nan)
        damping_uncertainties.append(np.nan)

# Plot results in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Simulation Time vs npart
for npart, data in aggregated_results.items():
    trials_simulation_time = data["simulation_time"]
    trials_npart = [npart] * trials  # Match npart to trials
    axes[0, 0].scatter(trials_npart, trials_simulation_time, alpha=0.2, color='grey', label="Individual Runs" if npart == 100 else "")  # Trial points
axes[0, 0].errorbar(npart_values, simulation_times, yerr=simulation_time_std, fmt='o', label="Mean Simulation Time", capsize=5)
axes[0, 0].set_title("Simulation Time vs npart")
axes[0, 0].set_xlabel("Number of Particles (npart)")
axes[0, 0].set_ylabel("Time (s)")
axes[0, 0].set_xscale("log")
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: RMS Noise vs npart
for npart, data in aggregated_results.items():
    trials_rms_noise = data["rms_noise"]
    trials_npart = [npart] * trials  # Match npart to trials
    axes[0, 1].scatter(trials_npart, trials_rms_noise, alpha=0.2, color='grey', label="Individual Runs" if npart == 100 else "")  # Trial points
axes[0, 1].errorbar(npart_values, rms_noise_values, yerr=rms_noise_std, fmt='o', label="Mean RMS Noise", capsize=5)
axes[0, 1].set_title("RMS Noise vs npart")
axes[0, 1].set_xlabel("Number of Particles (npart)")
axes[0, 1].set_ylabel("RMS Noise")
axes[0, 1].set_xscale("log")
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot 3: Harmonic Frequency vs npart
for npart, data in aggregated_results.items():
    if data["omega"]:
        trials_omega = data["omega"]
        trials_omega_unc = data["omega_uncertainty"]
        trials_npart = [npart] * len(trials_omega)  # Match npart to trials
        axes[1, 0].errorbar(trials_npart, trials_omega, yerr=trials_omega_unc, fmt='o', alpha=0.2, color='grey', label="Individual Runs" if npart == 100 else "")  # Trial points
axes[1, 0].errorbar(npart_values, omega_values, yerr=omega_uncertainties, fmt='o', label="Mean Harmonic Frequency (Omega)", capsize=5)
axes[1, 0].set_title("Harmonic Frequency vs npart")
axes[1, 0].set_xlabel("Number of Particles (npart)")
axes[1, 0].set_ylabel("Frequency (rad/s)")
axes[1, 0].set_xscale("log")
axes[1, 0].grid(True)
axes[1, 0].legend()

# Plot 4: Damping Rate vs npart
for npart, data in aggregated_results.items():
    if data["damping_rate"]:
        trials_damping_rate = data["damping_rate"]
        trials_damping_unc = data["damping_uncertainty"]
        trials_npart = [npart] * len(trials_damping_rate)  # Match npart to trials
        axes[1, 1].errorbar(trials_npart, trials_damping_rate, yerr=trials_damping_unc, fmt='o', alpha=0.2, color='grey', label="Individual Runs" if npart == 100 else "")  # Trial points
axes[1, 1].errorbar(npart_values, damping_rates, yerr=damping_uncertainties, fmt='o', label="Mean Damping Rate (Gamma)", capsize=5)
axes[1, 1].set_title("Damping Rate vs npart")
axes[1, 1].set_xlabel("Number of Particles (npart)")
axes[1, 1].set_ylabel("Damping Rate")
axes[1, 1].set_xscale("log")
axes[1, 1].grid(True)
axes[1, 1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()