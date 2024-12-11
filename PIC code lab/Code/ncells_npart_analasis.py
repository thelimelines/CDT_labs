from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from Electrostatic_PIC_1D import landau, run, Summary
from Harmonic_peak_plotter import analyze_harmonic_data
from numpy import linspace, pi
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Simulation parameters
L = 4 * pi  # Domain length
output_times = linspace(0, 20, 50)  # 50 output points between t=0 and t=20
trials = 5  # Number of trials per (ncells, npart) pair

# Ranges for parameters
npart_range = np.unique(np.logspace(2, 5, num=20, dtype=int))
ncells_range = range(10, 51, 5)

# Function to execute a single simulation
def run_simulation(ncells, npart, seed):
    np.random.seed(seed)
    pos, vel = landau(npart, L)

    summary = Summary()
    start_time = time.time()
    run(pos, vel, L, ncells=ncells, out=[summary], output_times=output_times)
    end_time = time.time()

    times = np.array(summary.t)
    harmonics = np.array(summary.firstharmonic)
    A0, d, A0_uncertainty, d_uncertainty, omega, omega_uncertainty, noise_rms, _ = analyze_harmonic_data(
        times, harmonics, plot=False
    )

    return {
        "ncells": ncells,
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
    delayed(run_simulation)(ncells, npart, seed)
    for ncells in ncells_range
    for npart in npart_range
    for seed in range(1, trials + 1)
)

# Aggregate results by (ncells, npart)
aggregated_results = {}
for r in results:
    key = (r["ncells"], r["npart"])
    if key not in aggregated_results:
        aggregated_results[key] = {
            "simulation_time": [],
            "rms_noise": [],
            "omega": [],
            "omega_uncertainty": [],
            "damping_rate": [],
            "damping_uncertainty": [],
        }
    aggregated_results[key]["simulation_time"].append(r["simulation_time"])
    aggregated_results[key]["rms_noise"].append(r["rms_noise"])
    if not np.isnan(r["omega"]) and not np.isnan(r["omega_uncertainty"]):
        aggregated_results[key]["omega"].append(r["omega"])
        aggregated_results[key]["omega_uncertainty"].append(r["omega_uncertainty"])
    if not np.isnan(r["damping_rate"]) and not np.isnan(r["damping_uncertainty"]):
        aggregated_results[key]["damping_rate"].append(r["damping_rate"])
        aggregated_results[key]["damping_uncertainty"].append(r["damping_uncertainty"])

# Compute means for aggregated data
ncells_vals, npart_vals, metrics = {}, {}, {}
metrics_list = ["simulation_time", "rms_noise", "omega", "damping_rate"]

for metric in metrics_list:
    ncells_vals[metric], npart_vals[metric], metrics[metric] = [], [], []
    for (ncells, npart), data in aggregated_results.items():
        ncells_vals[metric].append(ncells)
        npart_vals[metric].append(npart)
        if data[metric]:
            metrics[metric].append(np.mean(data[metric]))
        else:
            metrics[metric].append(np.nan)
    ncells_vals[metric] = np.array(ncells_vals[metric])
    npart_vals[metric] = np.array(npart_vals[metric])
    metrics[metric] = np.array(metrics[metric])

# Plot each metric in a separate 3D surface plot
for metric in metrics_list:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Reshape for surface plot
    unique_ncells = np.unique(ncells_vals[metric])
    unique_npart = np.unique(npart_vals[metric])
    X, Y = np.meshgrid(unique_npart, unique_ncells)
    Z = np.full(X.shape, np.nan)

    for i, ncells in enumerate(unique_ncells):
        for j, npart in enumerate(unique_npart):
            idx = (ncells_vals[metric] == ncells) & (npart_vals[metric] == npart)
            if np.any(idx):
                Z[i, j] = metrics[metric][idx][0]

    # Create surface plot
    surf = ax.plot_surface(
        np.log10(X), Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8
    )

    # Add labels and title
    ax.set_title(f"3D Surface Plot of {metric.replace('_', ' ').capitalize()} vs npart and ncells")
    ax.set_xlabel("Log(Number of Particles)")
    ax.set_ylabel("Number of Cells")
    ax.set_zlabel(f"{metric.replace('_', ' ').capitalize()}" + (" (unitless)" if metric == "rms_noise" else ""))

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    cbar.set_label(f"{metric.replace('_', ' ').capitalize()}")

    plt.show()