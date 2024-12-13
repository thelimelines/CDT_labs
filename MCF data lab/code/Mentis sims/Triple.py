import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import os

# Path to the folder containing the .mat files
folder_path = "MCF data lab/code/Mentis sims/Nbar-sweep"

# Define the parameters to extract and plot
parameters = {
    'ni0': {'label': r'$n_i$', 'unit': r'$(\mathrm{m}^{-3})$'},
    'te0': {'label': r'$T_e$', 'unit': r'(eV)'},
    'taue': {'label': r'$\tau_e$', 'unit': '(s)'}
}

# Initialize storage for results
nbi_powers = []
data_dict = {param: {'mean': [], 'std': []} for param in parameters}

def get_variable(data, index, subsection='zerod'):
    """
    Extract a variable from the dataset.
    """
    a = data['post'][subsection][0][0][index][0][0]
    return [float(x[0]) for x in a]

def get_average_and_std(data, start, end, index, subsection='zerod'):
    """
    Calculate mean and standard deviation for a given range.
    """
    variable = get_variable(data, index, subsection=subsection)
    subset = variable[start:end]
    return np.mean(subset), np.std(subset)

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

# Extract values for ni0, te0, and taue
ni_means = data_dict['ni0']['mean']
ni_stds = data_dict['ni0']['std']
te_means = data_dict['te0']['mean']
te_stds = data_dict['te0']['std']
tau_means = data_dict['taue']['mean']
tau_stds = data_dict['taue']['std']

# Calculate the triple product and propagate uncertainties
triple_product = ni_means * te_means * tau_means
triple_product_uncertainty = triple_product * np.sqrt(
    (ni_stds / ni_means)**2 + (te_stds / te_means)**2 + (tau_stds / tau_means)**2
)

# Plotting the triple product with uncertainties
plt.figure(figsize=(8, 6))
plt.plot(nbi_powers, triple_product, 'o-', label='Triple Product')
plt.fill_between(nbi_powers, triple_product + triple_product_uncertainty, triple_product - triple_product_uncertainty, alpha=0.2)
plt.xlabel(r"Line average electron density $(\times 10^{19} m^{-3})$")
plt.ylabel(r"Triple Product ($n_i T_e \tau_e$)")
plt.title("Triple Product vs. Nbar")
plt.grid(True)
plt.show()
