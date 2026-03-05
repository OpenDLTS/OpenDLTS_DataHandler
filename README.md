# OpenDLTS_DataHandler

Open source Deep-Level Transient Spectroscopy (DLTS) data processing program.

## Overview

**OpenDLTS_DataHandler** is a Python-based tool designed for the analysis of DLTS data. It utilizes `ipywidgets` for interactive data visualization and employs convex optimization techniques for advanced data fitting.

## Features

*   **Data Preprocessing**: Efficient handling of raw transient data, including outlier exclusion and signal cleaning.
*   **DLTS Spectrum Generation**: Supports traditional DLTS spectrum analysis using correlation function methods with customizable rate windows.
*   **LDLTS Algorithms**: Implements Laplace DLTS (LDLTS) using various inversion methods:
    *   L1 Regularization
    *   L2 Regularization (Tikhonov)
    *   D2 / Contin-like regularization
    *   **Sparse Representation** (Primary contribution of this work)

## Prerequisites

### Python Environment

You can set up the required Python environment using either Conda or Pip.

**Using Conda:**
```bash
conda env create -f environment.yml
```

**Using Pip:**
```bash
pip install -r requirements.txt
```

### Solvers

This project relies on robust convex optimization solvers. You must have a valid license for at least one of the following solvers:
*   **GUROBI**
*   **MOSEK**
*   **COPT**

## Examples

The example section is still under development.

## Citation

If you find this project useful in your research, please cite:

> **DOI**: 10.1109/TPEL.2026.3666365
