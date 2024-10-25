# MRI Simulation Toolkit

This repository provides scripts to simulate MRI data processing and viewing with GPU and CPU options. The toolkit includes:
- **`process_mri.py`**: A simulated MRI image processing script with CPU and optional GPU support.
- **`view_mri.py`**: A script to visualize MRI data in slices or 3D renderings.

## Requirements

- Python 3.8+
- `nibabel`, `numpy`, `cupy` (if using GPU), `scipy`, `psutil`, `pynvml` (for GPU utilization), `matplotlib` (for viewing)
  
Install requirements:
```bash
pip install nibabel numpy cupy scipy psutil pynvml matplotlib
```

**process_mri.py (MRI Data Processing)**
## Description

This script processes MRI images using different energy profiles. You can process data on the CPU by default or add GPU support for specific or all phases.

Usage
```bash
python process_mri.py <input_path> <output_path> [--multiplier <int>] [--scale_factor <float>] [--gpu] [--gpu-all]
<input_path>: Path to the input NIfTI file.
<output_path>: Path to save the processed file.
--multiplier: Optional slice expansion multiplier.
--scale_factor: Optional resizing scale factor.
--gpu: Use GPU for preprocessing.
--gpu-all: Use GPU for all phases.
```

Example
```bash
python process_mri.py input.nii output.nii --multiplier 2 --scale_factor 1.5 --gpu
```
**view_mri.py (MRI Data Viewing)**
## Description
This script displays MRI data either as slices or a 3D rendering. The viewer supports single-slice display, slice-by-slice animation, or full 3D volume rendering.

Usage
```bash
python view_mri.py <input_path> [--mode <render|slice>] [--slice_num <int>] [--roll]
<input_path>: Path to the input NIfTI file.
--mode: render for 3D or slice for 2D.
--slice_num: Specify slice number in slice mode.
--roll: Enable auto-scroll through slices with a 1-second delay.
```
Example

```bash
python view_mri.py input.nii --mode slice --slice_num 50
```

Ensure CUDA is installed for GPU options.