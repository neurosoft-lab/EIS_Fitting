# EIS Fitting

EIS Fitting fits DCH, untreated DCH, and ICH across Overlap, Length, Angle, and Width studies. Each study is fitted jointly across four independent devices. The desktop application provides single-fit and sequential batch-fit modes.

For each device, replicate 1 is selected prospectively (`ch1-1.txt` through `ch8-1.txt`). The other two scans remain technical quality-control repeats and are not counted as independent samples. Material parameters are shared across devices, each device has its own measurement-series resistance, and all four devices receive equal objective weight.

## Fixed model

- Measurement path: `Rs + Ztip + Znetwork`.
- Tip interface: `(Rt + semi-infinite Warburg) || CPE`, scaled by the physical distal end-face area.
- DCH and untreated DCH bulk: continuous PEDOT and ionic rails coupled by distributed `Zie` elements.
- DCH and untreated DCH overlap: equipotential Au, PEDOT, and ionic rails. Au couples to PEDOT through distributed `Rc`; PEDOT couples to the ionic rail through distributed `Zie`.
- ICH bulk: ionic rail only.
- ICH overlap: equipotential Au coupled directly to the ionic rail through distributed overlap elements.
- No direct Au-to-hydrogel branch is included for DCH or untreated DCH.
- No axial Au resistance, measurement-port capacitance, contact-layer correction, edge correction, taper correction, junction correction, or material-order constraint is included.

The distributed coupling element is

`Zie = [(Rct + Zw)^-1 + jw Cdl]^-1`.

The tip element is

`Ztip = {A_tip [(Rt + Zw,t)^-1 + Qt (jw)^nt]}^-1`.

## Concentration polarization

A lumped bulk concentration-polarization impedance is enabled for:

- untreated DCH Length
- ICH Length
- untreated DCH Angle
- ICH Angle

It is disabled for DCH and for all Overlap and Width studies.

## Use

Double-click `Launch.bat` in the package root for the desktop application. Choose a study and material, verify the material folder containing `Device1` through `Device4`, edit numerical parameter values or bounds if required, and select **Run**. For all studies, use **Batch fit**. Batch execution runs Overlap before Length, Angle, and Width and uses the current same-material four-device Overlap result in a fixed-then-bounded-release workflow.

Results are written under `03_Bin/Results/<study>/<material>/Fitting`. Completed Overlap fits replace the matching material reference bundle in `02_Reference/<material>/current`. Source EIS files are read only.

## Input files

Each material folder must contain `Device1/Rawdata_txt` through `Device4/Rawdata_txt`. Every device folder must contain `ch1-1.txt` through `ch8-1.txt`. Each file contains frequency, real impedance, and negative imaginary impedance columns.

## Outputs

Each completed result includes device-resolved fitted curves, an across-device mean ± s.d. Bode summary and Origin table, joint parameter estimates and bounds, device-resolved fit metrics, mesh convergence, network profiles, optimizer diagnostics, input hashes, and a prediction lock. Parameter uncertainty is deliberately not inferred from technical scans or the local optimizer covariance; use independent-device resampling or independent-device fits for confidence intervals.

## Package layout and installation

- `01_Data`: raw input files. The release contains no measurements.
- `02_Reference`: one current Overlap reference bundle per material. The release contains no fitted results.
- `03_Bin`: application, geometry definitions, dependencies list, and this guide.
- `Launch.bat`: desktop entry point; command-line arguments are also accepted.

Python 3.11 or later with Tk is required. Install dependencies with your chosen Python:

```text
python -m pip install -r 03_Bin/requirements.txt
```

The launcher checks installed Python environments. Set `EIS_PYTHON` to your preferred python.exe if required. A local environment at `03_Bin/.venv` is also supported. Dependency installation is manual.

Place input files at:

```text
01_Data/<study>/<material>/Device1/Rawdata_txt/ch1-1.txt
```

Use studies `1_Overlap`, `2_Length`, `3_Angle`, `4_Width`; materials `DCH`, `DCH (untreated)`, `ICH`; devices `Device1` through `Device4`; channels `ch1-1.txt` through `ch8-1.txt`. Each file has at least three rows with frequency (Hz), real impedance (ohm), and negative imaginary impedance (ohm). Headers are allowed. Run Overlap first to create references for the other studies. Single-fit mode also accepts a custom material folder via Browse, then Refresh. Parameter edits apply to single fits; batch fitting uses model defaults and the displayed search settings.

```text
Launch.bat single --study Overlap --material DCH --workers 4
Launch.bat batch
Launch.bat resume "path/to/result" --batches 1
```

The application opens without data. Refresh shows geometry and default parameters; Preview data and Run validate the selected data folders. Input files are read only. Fit checkpoints, diagnostics, and input hashes are created for new runs to support resume and reproducibility.

Before uploading a used copy, remove local measurements, reference results, `03_Bin/Results`, and any local `.venv`. Keep the empty data folders in Git using their `.gitkeep` files.

## Publishing on GitHub

Extract the distribution ZIP, then upload the contents of the `EIS_Fitting` folder to the repository root. Keep `01_Data`, `02_Reference`, `03_Bin`, and `Launch.bat` together. Include the `.gitkeep` files so the empty data directories are retained. The usage guide remains at `03_Bin/README.md` to preserve the required root layout.

The ZIP may also be attached to a GitHub release as a downloadable distribution. Uploading only the ZIP as a repository file stores the archive; extract it first to publish browsable source files.

This is a Python source distribution with a Windows launcher. Python, Tk, and the listed scientific dependencies must be installed separately. No local Python environment, measurements, or fitted reference data are bundled. `requirements.lock.txt` records the versions used for verification; `requirements.txt` specifies the minimum dependency versions.
