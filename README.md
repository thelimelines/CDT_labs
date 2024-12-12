# CDT_labs

This repository contains code, data, and analysis for computational labs conducted as part of the fusion CDT program under the university of York. The repository is organized into two main sections (thus far):

1. **PIC Code Lab**: Focuses on Particle-in-Cell (PIC) simulations to study Landau damping in 1D plasma systems.
2. **MCF Data Lab**: Contains data and tools for Magnetically Confined Fusion (MCF) simulations and analysis.

---

## **PIC Code Lab**

### **Overview**

The PIC Code Lab focuses on understanding Landau damping using 1D Particle-in-Cell simulations. The work explores how parameters such as the number of particles, grid cells, and system size influence the damping rate and frequency of plasma waves.

### **Folder Descriptions**

- **Code/**: Contains Python scripts for running PIC simulations and conducting various parameter analyses.
  - `Electrostatic_PIC_1D.py`: The main script for running 1D electrostatic PIC simulations.
  - `Harmonic_peak_plotter.py`: A tool for analyzing harmonic amplitudes and visualizing damping behavior.
  - `ncells_analysis_parallel.py`: Script for parameter sweeps over \( n_{\text{cells}} \) using parallelized execution.
  - `ncells_analysis_parallel_ensemble.py`: Extended parallel script for ensemble analyses over \( n_{\text{cells}} \).
  - `npart_analysis_parallel_ensemble.py`: Script for exploring the impact of varying \( n_{\text{part}} \) on simulation outcomes.
  - `L_analysis.py`: Analyzes the effect of varying system size (\( L \)) on Landau damping.

- **Data/**: Contains raw and processed simulation outputs stored as CSV files.
  - Files include descriptive names with timestamps, e.g., `landau_damping_harmonics_YYYYMMDDTHHMMSS.csv`.

---

## **Setup and Usage**

### **Dependencies**
- **Python Version**: 3.x
- **Required Libraries**:
  - `numpy`: For numerical computations.
  - `matplotlib`: For creating plots and visualizations.
  - `scipy`: For advanced numerical methods like curve fitting.
  - `pathlib`: For cross-platform file path handling.
  - `tkinter`: For GUI-based file selection during analysis.
- **Optional Libraries**:
  - `joblib`: For parallelizing simulation runs.

### **Installation**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PIC_code_lab
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy pathlib tkinter joblib
   ```

### **Running Simulations**
1. Run the core simulation:
   ```bash
   python Code/Electrostatic_PIC_1D.py
   ```
2. Perform parameter sweeps:
   - For \( n_{\text{cells}} \):
     ```bash
     python Code/ncells_analysis_parallel.py
     ```
   - For \( n_{\text{part}} \):
     ```bash
     python Code/npart_analysis_parallel_ensemble.py
     ```
3. Explore the effect of system size (\( L \)):
   ```bash
   python Code/L_analysis.py
   ```

### **Analyzing Results**
- Visualize harmonic data and analyze key parameters:
  ```bash
  python Code/Harmonic_peak_plotter.py
  ```
  
---


## **Acknowledgments**

This project is part of the CDT Computational Labs series. Special thanks to supervisors and lab coordinators for their guidance and support.

---

## **MCF Data Lab**

### **Overview**

The MCF Data Lab is another key section of this repository. It focuses on analyzing magnetically confined fusion (MCF) simulations and datasets. This section is under development, and updates will be provided as new tools and data become available.

---
