EIS FITTING - USER MANUAL
=========================

1. PURPOSE

EIS Fitting performs geometry-resolved electrochemical impedance spectroscopy
fitting for DCH, untreated DCH, and ICH in Overlap, Length, Angle, and Width
studies. It supports single-study fitting and sequential batch fitting.

The standard workflow jointly fits four independent devices using replicate 1
from each channel. Material parameters are shared across devices; measurement
series resistance is device-specific. Technical repeats are not independent
samples. Raw input files are read only.

2. PACKAGE CONTENTS

01_Data       Raw EIS input data. Empty in the source distribution.
02_Reference  Current fitted Overlap references, separated by material.
              Empty in the source distribution.
03_Bin        Program source, geometry definitions, dependency lists,
              detailed model documentation, and generated fitting results.
Launch.bat    Windows application launcher.
Readme.txt    This user manual.

Keep these items together when moving the application. The .gitkeep files in
the data directories retain the empty folders when publishing the source.

3. REQUIREMENTS AND INSTALLATION

Use Windows with Python 3.11 or later, including Tkinter. The scientific
dependencies are NumPy, SciPy, pandas, and Matplotlib. SciPy 1.16 or later is
required for the optimizer callback used by pause and stop controls.

Open a terminal in the EIS_Fitting folder and run:

    python -m pip install -r 03_Bin/requirements.txt

Use the Python executable for your intended environment if the command above
selects a different installation. The launcher checks available Python
environments. To select one explicitly, set EIS_PYTHON to its python.exe path.
A local environment at 03_Bin/.venv is also supported.

03_Bin/requirements.txt specifies minimum versions.
03_Bin/requirements.lock.txt records the versions used for verification.
The distribution does not include Python or install dependencies automatically.

4. PREPARE RAW DATA

Place data under the following hierarchy:

    01_Data/<study>/<material>/Device1/Rawdata_txt/ch1-1.txt

Study folder names:
    1_Overlap
    2_Length
    3_Angle
    4_Width

Material folder names:
    DCH
    DCH (untreated)
    ICH

Each material folder must contain Device1 through Device4. Each device must
contain a Rawdata_txt folder with ch1-1.txt through ch8-1.txt.

Each file must contain at least three numeric rows. The first three columns are:
    Frequency in Hz; real impedance in ohm; negative imaginary impedance in ohm.

Use whitespace or commas as separators. Header rows are allowed. Frequencies
must be positive and unique, and numeric values must be finite. Use the same
frequency grid across devices for each channel to enable device summary plots.

The supplied geometry definitions are in 03_Bin/Templates/catalog.json. Verify
that the specimen dimensions match these definitions before fitting new data.

5. START AND RUN A SINGLE FIT

1) Double-click Launch.bat. The interface can open before data are provided.
2) Select the study and material in the Single fit tab.
3) Confirm the Data folder. For a custom location, use Browse to select the
   material folder containing Device1 through Device4, then click Refresh.
4) Click Preview data to inspect the measured spectra.
5) Double-click a parameter to edit its initial value and bounds if needed.
6) Set the search settings and click Run.
7) Inspect the Results and Log tabs. Use Pause, Resume, or Stop as needed.
   Stopping waits for a safe checkpoint and result export; it is not immediate.

Fit Overlap first for each material. Length, Angle, and Width use the matching
material's Overlap reference in a fixed-then-bounded-release workflow.

6. BATCH FITTING AND REFERENCES

In Batch fit, double-click a task to switch its Run setting between Yes and No.
Click Run selected after confirming that the selected tasks have input data.
The default order is Overlap, Length, Angle, then Width, for all materials.
Batch fitting uses model default parameters and the displayed search settings;
parameter edits in Single fit apply to single fits.

Completed Overlap fits automatically save a complete reference bundle to:

    02_Reference/<material>/current

A new registered Overlap result replaces that material's current reference.
The application creates the reference from an actual fitted result; no
measurement or reference values are supplied with the source distribution.

7. RESULTS AND INTERPRETATION

New fitting results are written under:

    03_Bin/Results/<study>/<material>/Fitting

Result packages include fitted curves, Bode and Nyquist plots, Origin tables,
device-level fit metrics, fitted parameters and bounds, mesh convergence,
network profiles, optimizer diagnostics, input hashes, and resume checkpoints.
The interface can reopen a completed result using Open result.

Check residuals, parameter bounds, mesh convergence, and network diagnostics
before interpreting fitted values. Optimizer convergence does not prove a
global optimum or uniquely identified physical parameters. Confidence intervals
are not inferred from technical repeats or local optimizer covariance.

Detailed model equations and assumptions are in 03_Bin/README.md.

8. OPTIONAL COMMAND-LINE USE

Run these commands from the EIS_Fitting folder:

    Launch.bat single --study Overlap --material DCH --workers 4
    Launch.bat single --study Overlap --material "DCH (untreated)"
    Launch.bat batch
    Launch.bat resume "path/to/completed/result" --batches 1

For available options:

    Launch.bat single --help
    Launch.bat batch --help
    Launch.bat resume --help

Resume requires unchanged model code, input data, geometry, parameter bounds,
and the fitting settings included in the checkpoint signature.

9. TROUBLESHOOTING

No compatible Python environment:
    Install the dependencies in the intended Python environment. Check its
    SciPy version and Tkinter availability, or set EIS_PYTHON explicitly.

Missing device folder or channel file:
    Check the folder names and required Device1-Device4 / CH1-CH8 structure.
    After browsing to another material folder, click Refresh.

Missing reference:
    Complete an Overlap fit for the same material before another study.

Reference rejected:
    Use a complete matching-material, four-device Overlap result from the
    current model. The staged workflow requires passing mesh convergence.

Resume rejected:
    Restore the original data and fitting configuration or start a new fit.

10. PUBLISHING THE SOURCE

Upload 01_Data, 02_Reference, 03_Bin, Launch.bat, and Readme.txt together to the
GitHub repository root. Keep the .gitkeep files. Extract the ZIP before uploading
if you want the source files to be browsable in the repository. The ZIP can also
be attached to a GitHub release for download.

Before publishing a copy that has been used for fitting, remove local raw data,
reference results, 03_Bin/Results, local environments, and runtime caches. The
clean source distribution contains no experimental measurements or fit results.
