import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import os

# Path to the folder containing the .mat files
folder_path = "MCF data lab/code/Mentis sims/Nbar-sweep"

# Define the parameters to plot
parameters = {
    'ni0': {'label': r'$n_i$', 'unit': r'$(\mathrm{m}^{-3})$'},
    'ne0': {'label': r'$n_e$', 'unit': r'$(\mathrm{m}^{-3})$'},
    'te0': {'label': r'$T_e$', 'unit': r'(eV)'},
    'taue': {'label': r'$\tau_e$', 'unit': '(s)'},
    'betap': {'label': r'$\beta_p$', 'unit': '(unitless)'}}#,
    #'modeh': {'label': r'$mode$', 'unit': '(h-mode 1, l-mode 0)'}
#}

# Initialize storage for results
nbi_powers = []
data_dict = {param: {'mean': [], 'std': []} for param in parameters}

def get_variable(data, index, subsection='zerod'):
    a = data['post'][subsection][0][0][index][0][0]
    return [float(x[0]) for x in a]

def get_average_and_std(data, start, end, index, subsection='zerod'):
    variable = get_variable(data, index, subsection=subsection)
    subset = variable[start:end]
    return np.mean(subset), np.std(subset)

# Initialize storage for results
nbi_powers = []
data_dict = {param: {'mean': [], 'std': []} for param in parameters}

# Process each .mat file
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith(".mat"):
        # Extract NBI power from the filename
        power = float(file_name.split('_')[1].replace('.mat', ''))
        nbi_powers.append(power)
        
        # Load the .mat file
        file_path = os.path.join(folder_path, file_name)
        full_dataset = scipy.io.loadmat(file_path)
        
        # Extract and store the mean and std for each parameter in the pulse range
        for param in parameters:
            mean, std = get_average_and_std(full_dataset, 50, 100, param)
            data_dict[param]['mean'].append(mean)
            data_dict[param]['std'].append(std)

# Sort the data by NBI powers
sorted_indices = np.argsort(nbi_powers)
nbi_powers = np.array(nbi_powers)[sorted_indices]  # Sort NBI powers
for param in parameters:
    data_dict[param]['mean'] = np.array(data_dict[param]['mean'])[sorted_indices]
    data_dict[param]['std'] = np.array(data_dict[param]['std'])[sorted_indices]

# Create vertically stacked subplots with error bars
fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12), sharex=True)

for i, param in enumerate(parameters):
    means = data_dict[param]['mean']
    stds = data_dict[param]['std']
    label = f"{parameters[param]['label']} {parameters[param]['unit']}"  # Label with unit
    axes[i].plot(nbi_powers, means, 'o-', label=label)
    axes[i].set_ylabel(label)
    axes[i].fill_between(nbi_powers, means + stds, means - stds, alpha=0.2)
    axes[i].grid()

# Shared x-axis label
axes[-1].set_xlabel(r"Line average electron density $(\times 10^{19} m^{-3})$")

# Adjust layout
plt.subplots_adjust(hspace=0)
plt.show()