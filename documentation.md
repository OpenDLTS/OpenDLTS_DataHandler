# OpenDLTS_DataHandler API Documentation

**OpenDLTS_DataHandler** is a comprehensive Python toolkit for Deep-Level Transient Spectroscopy (DLTS) data analysis. It provides tools for data preprocessing, traditional DLTS spectrum generation, and advanced Laplace DLTS (LDLTS) analysis using convex optimization techniques.

---

## Table of Contents

- [Core Data Classes](#core-data-classes)
  - [Data_Loader](#data_loader)
  - [Trap](#trap)
  - [Material](#material)
  - [DLTS_Data_Generator](#dlts_data_generator)
- [DLTS Correlation Functions](#dlts-correlation-functions)
  - [DLTS_CORRELATION_FUNCTION](#dlts_correlation_function)
- [LDLTS Methods](#ldlts-methods)
  - [L1 (L1 Regularization)](#l1-regularization)
  - [L2 (L2 Regularization)](#l2-regularization)
  - [D2 (Contin-like Regularization)](#d2-regularization)
  - [SR (Sparse Representation)](#sparse-representation)
  - [SRL1 (SR + L1 Hybrid)](#srl1-hybrid)
- [Widgets (Interactive Visualization)](#widgets)
- [Examples](#examples)

---

## Core Data Classes

### Data_Loader

**`class OpenDLTS_DataHandler.Data_Loader`**

Core class for handling DLTS experimental datasets. Provides data loading, preprocessing, resampling, and export functionality.

#### Initialization

```python
Data_Loader(
    transient_data: str | Path | dict | np.ndarray,
    raw_data_scaling_factor: float = 1.0,
    time_shift: float | np.ndarray = 0,
    T_shift: float | np.ndarray = 0,
    data_scaling_factor: float = 1e12,
    data_type: str = 'C',
    data_x_type: str = 'Time',
    condition_type: str = 'Temperature',
    logging_level: str = 'info',
    logging_file: str | Path | None = None,
    logging_file_clear: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transient_data` | `str \| Path \| dict \| ndarray` | - | Input data: file path, dictionary `{'t', 'T', 'C'}`, or 2D array |
| `raw_data_scaling_factor` | `float` | `1.0` | Scaling factor for raw input data |
| `time_shift` | `float \| ndarray` | `0` | Time axis shift [s] |
| `T_shift` | `float \| ndarray` | `0` | Temperature axis shift [K] |
| `data_scaling_factor` | `float` | `1e12` | Final data scaling (e.g., F→pF) |
| `data_type` | `str` | `'C'` | Data type: `'C'` (capacitance), `'I'` (current), `'V'` (voltage) |
| `data_x_type` | `str` | `'Time'` | X-axis type |
| `condition_type` | `str` | `'Temperature'` | Condition variable type |
| `logging_level` | `str` | `'info'` | Logging level |
| `logging_file` | `str \| Path \| None` | `None` | Log file path |
| `logging_file_clear` | `bool` | `False` | Clear log file on init |

**Attributes (after initialization):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `t` | `ndarray` | Time measurement points [s] |
| `T` | `ndarray` | Temperature/condition values |
| `C` | `ndarray` | Signal data matrix (shape: `[n_conditions, n_times]`) |
| `data_plot_dict` | `dict` | Plot configuration (labels, units, scaling) |
| `rawdata` | `ndarray` | Original raw data in DLTS format |

#### Methods

##### `data_space(num, space='log', lin_interp=False)`

Resample data points in linear or logarithmic time space.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num` | `int` | - | Number of points in new time axis |
| `space` | `str` | `'log'` | `'lin'` (linear) or `'log'` (logarithmic) |
| `lin_interp` | `bool` | `False` | Use linear interpolation if `True` |

**Returns:** `Data_Loader` - New instance with resampled data

**Example:**
```python
loader_resampled = loader.data_space(num=500, space='log')
```

---

##### `savedata(newdatafile, ignore_scaling=False)`

Export data in DLTS format.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `newdatafile` | `str \| Path` | `'temp_output_data.transdata'` | Output filename |
| `ignore_scaling` | `bool` | `False` | If `True`, omit final unit scaling |

**Example:**
```python
loader.savedata('output_data.transdata', ignore_scaling=True)
```

---

##### `copy()`

Create a deep copy of the Data_Loader instance.

**Returns:** `Data_Loader`

---

##### Operator Overloads

| Operator | Description |
|----------|-------------|
| `loader1 - loader2` | Subtract two Data_Loader instances (C = C₁ - C₂) |
| `-loader` | Negate signal (C = -C) |
| `loader1 + loader2` | Add two Data_Loader instances |
| `loader1 & loader2` | Merge datasets based on temperature axis |

---

### Trap

**`class OpenDLTS_DataHandler.Trap`**

Trap model for DLTS analysis. Supports Arrhenius traps, constant emission rate traps, and user-defined traps.

#### Initialization

```python
Trap(
    Ea: float | None = None,
    sigma: float | None = None,
    T_power: float | int = 2,
    constant_em: float | None = None,
    fun_amplitude_T: Callable | float | int = 1,
    fun_em_T: Callable | None = None,
    material: str = 'Si',
    material_doping_type: str = 'N',
    trap_type: str = 'majority'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Ea` | `float \| None` | `None` | Activation energy [eV] (Arrhenius trap) |
| `sigma` | `float \| None` | `None` | Capture cross-section [cm²] (Arrhenius trap) |
| `T_power` | `float \| int` | `2` | Temperature power factor (Arrhenius trap) |
| `constant_em` | `float \| None` | `None` | Constant emission rate [s⁻¹] (Constant emission rate trap)|
| `fun_amplitude_T` | `Callable \| float` | `1` | Amplitude vs temperature function |
| `fun_em_T` | `Callable \| None` | `None` | User-defined emission rate function (User-defined trap) |
| `material` | `str` | `'Si'` | Material name (`'Si'`, `'SiC'`, `'GaN'`) |
| `material_doping_type` | `str` | `'N'` | Doping type (`'N'` or `'P'`) |
| `trap_type` | `str` | `'majority'` | Trap type (`'majority'` or `'minority'`) |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `mat` | `Material` | Material object |
| `mat_dop_type` | `str` | Doping type |
| `trap_type` | `str` | Trap type |
| `em` | `Callable` | Emission rate function `em(T)` |
| `tau` | `Callable` | Time constant function `tau(T) = 1/em(T)` |
| `amp` | `Callable` | Amplitude function `amp(T)` |

#### Methods

##### `get_T_from_fixed_em(fixed_em, opt_bounds=(0.1, 10000))`

Get temperature from a fixed emission rate.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_em` | `float` | - | Target emission rate [s⁻¹] |
| `opt_bounds` | `tuple` | `(0.1, 10000)` | Temperature optimization bounds [K] |

**Returns:** `float | None` - Temperature [K]

---

##### `get_T_from_fixed_tau(fixed_tau, opt_bounds=(0.1, 10000))`

Get temperature from a fixed time constant.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_tau` | `float` | - | Target time constant [s] |
| `opt_bounds` | `tuple` | `(0.1, 10000)` | Temperature optimization bounds [K] |

**Returns:** `float | None` - Temperature [K]

---

#### Example

```python
from OpenDLTS_DataHandler import Trap

# Arrhenius trap: Ea=0.5 eV, sigma=1e-15 cm²
trap = Trap(Ea=0.5, sigma=1e-15, material='Si', material_doping_type='N')

# Get emission rate at 300K
em_300K = trap.em(300)

# Get temperature for emission rate = 1000 s⁻¹
T_at_1000 = trap.get_T_from_fixed_em(1000)
```

---

### Material

**`class OpenDLTS_DataHandler.Material`**

Material properties calculator for semiconductor materials. Provides temperature-dependent material parameters.

**Available Materials:**

| Material | Class Names |
|----------|-------------|
| Silicon | `Material.Si`, `Material.Silicon`, `Material.si`, `Material.silicon` |
| Silicon Carbide | `Material.SiC`, `Material.SiC4H`, `Material.SiliconCarbide` |
| Gallium Nitride | `Material.GaN`, `Material.GalliumNitride`, `Material.gan` |

#### Usage

```python
from OpenDLTS_DataHandler import Material

# Create Silicon material at 400K
si = Material.Si(T=400)

# Access properties
Eg = si.Eg          # Band gap [eV]
Nc = si.Nc          # Conduction band DOS [cm⁻³]
Nv = si.Nv          # Valence band DOS [cm⁻³]
vth_n = si.vth_n    # Electron thermal velocity [cm/s]
vth_p = si.vth_p    # Hole thermal velocity [cm/s]
ni = si.ni          # Intrinsic carrier concentration [cm⁻³]

# Print all parameters
si.print()
```

**Properties:**

| Property | Unit | Description |
|----------|------|-------------|
| `Eg` | `eV` | Band gap energy |
| `mnco` | `-` | Electron effective mass coefficient |
| `mn` | `kg` | Electron effective mass |
| `Nc` | `cm⁻³` | Conduction band density of states |
| `mpco` | `-` | Hole effective mass coefficient |
| `mp` | `kg` | Hole effective mass |
| `Nv` | `cm⁻³` | Valence band density of states |
| `vth_n` | `cm/s` | Electron thermal velocity |
| `vth_p` | `cm/s` | Hole thermal velocity |
| `ni` | `cm⁻³` | Intrinsic carrier concentration |

---

### DLTS_Data_Generator

**`class OpenDLTS_DataHandler.DLTS_Data_Generator`**

Generate simulated DLTS capacitance transients based on trap parameters.

#### Initialization

```python
DLTS_Data_Generator(trap_list: list[Trap])
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trap_list` | `list[Trap]` | List of Trap objects |

#### Methods

##### `get_data(T, t, sigma=0.2, DC_fun=lambda T: 200, savefile=None)`

Generate simulated DLTS data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T` | `float \| ndarray` | - | Temperature array [K] |
| `t` | `float \| ndarray` | - | Time array [s] |
| `sigma` | `float` | `0.2` | Noise standard deviation |
| `DC_fun` | `Callable` | `lambda T: 200` | DC capacitance vs temperature |
| `savefile` | `str \| Path \| None` | `None` | Output file path (optional) |

**Returns:** `ndarray` - DLTS data matrix in standard format

#### Example

```python
from OpenDLTS_DataHandler import DLTS_Data_Generator, Trap

# Define traps
trap1 = Trap(Ea=0.5, sigma=1e-15, material='Si')
trap2 = Trap(Ea=0.7, sigma=5e-14, material='Si')

# Create generator
generator = DLTS_Data_Generator([trap1, trap2])

# Generate data
T_range = np.linspace(100, 500, 50)
t_range = np.logspace(-6, -2, 200)
data = generator.get_data(T_range, t_range, sigma=0.1)
```

---

## DLTS Correlation Functions

### DLTS_CORRELATION_FUNCTION

**`class OpenDLTS_DataHandler.DLTS_CORRELATION_FUNCTION`**

Container class for various DLTS correlation function implementations. Each correlation function is optimized for detecting specific emission rate ranges.

**Reference:** [10.1063/1.1148038](https://doi.org/10.1063/1.1148038)

#### Available Correlation Functions

| Class | Order | Optimum td/tc | Optimum SNR | Description |
|-------|-------|---------------|-------------|-------------|
| `shifted_exponential` | 1 | 0.082 | 0.21 | Standard shifted exponential |
| `double_boxcar` | 1 | 0.131 | 0.13 | Double boxcar weighting |
| `triangular` | 2 | 0.037 | 0.092 | Triangular (2nd order) |
| `HiRes_4` | 3 | 0.011 | 0.029 | High resolution (3rd order) |
| `HiRes_5` | 4 | 0.007 | 0.013 | High resolution (4th order) |
| `HiRes_6` | 5 | 0.005 | 0.0058 | High resolution (5th order) |

#### Common Interface

All correlation function classes share the following interface:

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `optimum_td_tc_ratio` | `float` | Optimal time delay to time constant ratio |
| `optimum_pw` | `float` | Optimal pulse width setting |
| `optimum_SNR` | `float` | Optimal signal-to-noise ratio |
| `order` | `int` | Order of the correlation function |

**Methods:**

##### `main_fun(t, td, tc)`

Core correlation function calculation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `ndarray` | Time values array |
| `td` | `float` | Time delay parameter |
| `tc` | `float` | Time constant parameter |

**Returns:** `ndarray` - Weight function values

---

##### `find_em(tc0, tc1, em0=1e-1, em1=1e5, rel_tol=1e-3)`

Find optimal emission rate that maximizes the correlation signal.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tc0` | `float` | - | Start time for rate window [s] |
| `tc1` | `float` | - | End time for rate window [s] |
| `em0` | `float` | `1e-1` | Lower bound for emission rate [s⁻¹] |
| `em1` | `float` | `1e5` | Upper bound for emission rate [s⁻¹] |
| `rel_tol` | `float` | `1e-3` | Relative tolerance |

**Returns:** `tuple` - `(em_opt, em_list, val_list)`

---

##### `__call__(t, T, C, use_opt_ratio=True, ...)`

Compute DLTS signal for multiple temperature points.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t` | `ndarray` | - | Time measurement points |
| `T` | `ndarray` | - | Temperature values |
| `C` | `ndarray` | - | Capacitance transients matrix |
| `use_opt_ratio` | `bool` | `True` | Use optimal time ratio |
| `tc0` | `float` | `-1` | Manual start time |
| `tc1` | `float` | `-1` | Manual end time |
| `interp_type` | `str` | `'lin'` | `'lin'` or `'log'` |

**Returns:** `tuple` - `(T, val_list, rate_window)`

#### Example

```python
from OpenDLTS_DataHandler import DLTS_CORRELATION_FUNCTION

# Use shifted exponential correlation function
corr_func = DLTS_CORRELATION_FUNCTION.shifted_exponential()

# Compute DLTS spectrum
T_dlts, signal, rate_window = corr_func(t, T, C, use_opt_ratio=True)
```

---

## LDLTS Methods

All LDLTS methods inherit from a common base class and share similar interfaces. They implement Laplace DLTS analysis using convex optimization.

### L1 Regularization

**`class OpenDLTS_DataHandler.LDLTS_Method.L1`**

L1-regularized inverse problem solver for LDLTS analysis. Promotes sparse solutions in the emission rate spectrum.

L1 method solving problem:
```math
\mathrm{\mathop{arg\ min}\limits_{\mathbf{F}}\ \left\{ \left|\left|\mathbf{I}-\mathbf{A}\times\mathbf{F}\right|\right|_2^2 +\lambda \cdot \left|\left|\mathbf{F}\right|\right|_1 \right\}}
```

#### Initialization

```python
L1(dlts_data: Data_Loader, Ns: int = 500, s0: float = 1e-1, s1: float = 1e5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dlts_data` | `Data_Loader` | - | Input DLTS data |
| `Ns` | `int` | `500` | Number of emission rate points |
| `s0` | `float` | `1e-1` | Minimum emission rate [s⁻¹] |
| `s1` | `float` | `1e5` | Maximum emission rate [s⁻¹] |

#### Methods

##### `solve(Ti_list, polarity='positive', lambda1=1e-5, ...)`

Solve L1-regularized inverse problem.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Ti_list` | `ndarray` | - | Temperature indices to process |
| `polarity` | `str` | `'positive'` | `'positive'`, `'negative'`, or `'both'` |
| `verbose` | `bool` | `False` | Show solver output |
| `lambda1` | `float` | `1e-5` | Regularization parameter |
| `skip_solved` | `bool` | `True` | Skip if already solved |
| `enable_irls_mode` | `bool` | `False` | Enable IRLS mode |
| `solver` | `str` | `'CVXOPT'` | CVXPY solver name |
| `f_scaling_by_C` | `bool` | `False` | Scale by averaged capacitance |
| `constraint_max_rms` | `bool` | `True` | Constrain maximum RMS |

**Returns:** `None | dict` - Solution dictionary (if IRLS mode)

**Example:**
```python
from OpenDLTS_DataHandler import L1

ldlts = L1(loader, Ns=500)
ldlts.solve(Ti_list=np.arange(10), lambda1=1e-4)
```

---

##### `irls(solve_index=-1, irls_max_iter=20, irls_p_norm=0.5, ...)`

Perform Iteratively Reweighted Least Squares optimization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solve_index` | `int` | `-1` | Solution index to process |
| `irls_max_iter` | `int` | `20` | Maximum IRLS iterations |
| `irls_p_norm` | `float` | `0.5` | L-p norm exponent |
| `irls_rel_tol` | `float` | `1e-7` | Relative error tolerance |
| `verbose` | `bool` | `False` | Print iteration details |

**Returns:** `None`

**Updates:** `irls_solve_history` attribute

---

##### `lcurve(solve_arg={}, ramp_param_name='lambda1', ...)`

Perform L-curve regularization parameter optimization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solve_arg` | `dict` | `{}` | Solve parameters |
| `ramp_param_name` | `str` | `'lambda1'` | Parameter to optimize |
| `lcurve_init_lambda` | `float` | `1e-8` | Initial regularization value |
| `lcurve_max_attempt` | `int` | `20` | Maximum search iterations |

**Returns:** `bool` - `False` if converged

**Updates:** `lcurve_history` attribute

---

##### `prsse(solve_index=-1, si_list=None, fwhm_height=0.5)`

Perform Perturbation Response Spectral Sensitivity Estimation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solve_index` | `int` | `-1` | Solution index to process |
| `si_list` | `list \| None` | `None` | Emission rate indices |
| `fwhm_height` | `float` | `0.5` | FWHM height fraction |

**Returns:** `None`

**Updates:** `prsse_solve_history` attribute

---

##### `save_result(filepath=None, save_problem=True)` / `load_result(filepath=None)`

Save/load solver state to/from file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str \| Path \| None` | `None` | File path (auto-generated if None) |
| `save_problem` | `bool` | `True` | Include problem data |

---

### L2 Regularization

**`class OpenDLTS_DataHandler.LDLTS_Method.L2`**

L2-regularized (Tikhonov) inverse problem solver. Produces smooth emission rate spectra.

L2 method solving problem:
```math
\mathrm{\mathop{arg\ min}\limits_{\mathbf{F}}\ \left\{ \left|\left|\mathbf{I}-\mathbf{A}\times\mathbf{F}\right|\right|_2^2 +\lambda \cdot \left|\left|\mathbf{F}\right|\right|_2^2 \right\}}
```


**Interface:** Same as `L1`

**Key Difference:** Uses L2 norm regularization instead of L1, resulting in smoother solutions.

---

### D2 Regularization

**`class OpenDLTS_DataHandler.LDLTS_Method.D2`**

Second-derivative regularization (Contin-like). Produces very smooth emission rate spectra similar to the CONTIN algorithm.

D2 method solving problem:
```math
\mathrm{\mathop{arg\ min}\limits_{\mathbf{F}}\ \left\{ \left|\left|\mathbf{I}-\mathbf{A}\times\mathbf{F}\right|\right|_2^2 +\lambda \cdot \left|\left|\frac{d\mathbf{F}}{dt}\right|\right|_2^2 \right\}}
```

**Interface:** Same as `L1`

**Key Difference:** Uses second-derivative regularization for maximum smoothness.

---

### Sparse Representation

**`class OpenDLTS_DataHandler.LDLTS_Method.SR`**

Sparse representation solver using dictionary learning. Represents the emission rate spectrum as a combination of predefined basis functions (word functions).

SR method solving problem:
```math
\begin{equation}
\begin{aligned}
\begin{matrix}
    \mathrm{\mathop{arg\ min}\limits_{\mathbf{F},\ \mathbf{X}}}&\mathrm{\ \left\{ \left|\left|\mathbf{I}-\mathbf{A}\times\mathbf{F}\right|\right|_2^2 +\lambda \left|\left|\mathbf{R}\right|\right|_{1} \right\}}\\
    \mathrm{s.t.}&\mathrm{\ \mathbf{F^*}=\mathbf{D}\times\mathbf{X}}
\end{matrix}
\end{aligned}
\end{equation}
```

#### Additional Methods

##### `generate_dictionary_stack_sparse(word_fun_list, enumerate_args_list_dic_list, ...)`

Generate a stacked sparse dictionary matrix from multiple word functions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `word_fun_list` | `list` | Word function names or objects |
| `enumerate_args_list_dic_list` | `list` | Enumerable arguments for each function |
| `unenumerate_args_list_dic_list` | `list` | Common arguments for all functions |
| `polarity_list` | `list` | Polarity constraints |
| `dc_mono_list` | `list` | Monotonicity constraints |

**Returns:** `tuple` - `(D_matrix, info_list)`

---

##### `get_dictionary_from_word_sparse(word_fun, enumerate_args_list_dic, ...)`

Generate sparse dictionary for a single word function.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `word_fun` | `Callable \| str` | Word function |
| `enumerate_args_list_dic` | `dict` | Enumerable arguments |
| `unenumerate_args_list_dic` | `dict` | Common arguments |
| `polarity` | `str` | Polarity constraint |
| `dc_mono` | `str` | Monotonicity constraint |

**Returns:** `tuple` - `(D_matrix, info)`

---

### SRL1 Hybrid

**`class OpenDLTS_DataHandler.LDLTS_Method.SRL1`**

Hybrid solver combining Sparse Representation with L1 regularization. Uses dictionary-based representation for the main spectrum with L1 regularization for residual features.

SRL1 method solving problem:
```math
\begin{equation}
\begin{aligned}
\begin{matrix}
    \mathrm{\mathop{arg\ min}\limits_{\mathbf{F},\ \mathbf{X}}}&\mathrm{\ \left\{ \left|\left|\mathbf{I}-\mathbf{A}\times\mathbf{F}\right|\right|_2^2 +\lambda \left|\left|\mathbf{F^{\#}}\right|\right|_{1} \right\}}\\
    \mathrm{s.t.}&\mathrm{\ \mathbf{F^*}=\mathbf{D}\times\mathbf{X}}+\mathbf{F^{\#}}
\end{matrix}
\end{aligned}
\end{equation}
```

**Interface:** Extends `SR` with additional L1 regularization on residual component.

---

## Widgets

**`module OpenDLTS_DataHandler.Widgets`**

Interactive visualization components based on `ipywidgets` for Jupyter notebooks.

### Available Widgets

| Widget | Description |
|--------|-------------|
| `Ti_List_Selector` | Interactive temperature index selector |
| `Data_Viewer_Box` | Raw data viewer with zoom and pan |
| `DLTS_Viewer_Box` | Traditional DLTS spectrum viewer |
| `LDLTS_Viewer_Box` | LDLTS result viewer |
| `ARRH_PLOTTER` | Arrhenius plot generator |
| `TRANS_PLOTTER` | Transient curve plotter |
| `LDLTS_PLOTTER` | LDLTS spectrum plotter |
| `LDLTS_T_PLOTTER` | Temperature-dependent LDLTS plotter |

### Example Usage

```python
from OpenDLTS_DataHandler import Widgets, Data_Loader

# Load data
loader = Data_Loader('data.transdata')

# Create interactive viewer
viewer = Widgets.Data_Viewer_Box(loader)
viewer.display()

# Create Arrhenius plotter
arrh_plotter = Widgets.ARRH_PLOTTER(ldlts_result)
arrh_plotter.display()
```

---

## Dependencies

- **Core:** `numpy`, `scipy`, `cvxpy`
- **Visualization:** `ipywidgets`, `matplotlib`
- **Solvers:** Requires at least one of: `GUROBI`, `MOSEK`, or `COPT` (need licenses)

## Citation

If you use OpenDLTS_DataHandler in your research, please cite:

> **DOI:** [10.1109/TPEL.2026.3666365](https://doi.org/10.1109/TPEL.2026.3666365)

