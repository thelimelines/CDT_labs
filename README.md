# CDT_labs

This repository contains code, data, and analysis for computational labs conducted as part of the fusion CDT program under the university of York. The repository is organized into two main sections (thus far):

1. **PIC Code Lab**: Focuses on Particle-in-Cell (PIC) simulations to study Landau damping in 1D plasma systems.
2. **MCF Data Lab**: Contains data and tools for Magnetically Confined Fusion (MCF) simulations and analysis.

---

## **PIC Code Lab**

### **Overview**

The PIC Code Lab focuses on understanding Landau damping using 1D Particle-in-Cell simulations. The work explores how parameters such as the number of particles (\(n_{\text{part}}\)), grid cells (\(n_{\text{cells}}\)), and system size (\(L\)) influence the damping rate (\(\gamma\)) and frequency (\(\omega\)) of plasma waves.

### **Repository Structure**

```plaintext
PIC_code_lab/
├── Code/
│   ├── Electrostatic_PIC_1D.py
│   ├── Harmonic_peak_plotter.py
│   ├── ncells_analysis_parallel.py
│   ├── ncells_analysis_parallel_ensemble.py
│   ├── npart_analysis_parallel_ensemble.py
│   ├── L_analysis.py
│   ├── ...
├── Data/
│   ├── landau_damping_harmonics_<timestamp>.csv
│   ├── ...
├── Plots/
│   ├── phase_space_t<step>.png
│   ├── harmonic_decay_plot.png
│   ├── ...
├── Docs/
│   ├── lab_notes.pdf
│   ├── README.md
MCF_Data_Lab/
README
```

