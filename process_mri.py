import nibabel as nib
import numpy as np
import cupy as cp
import time
import os
from scipy import ndimage
from scipy.ndimage import zoom
import sys
import psutil
import argparse
import pynvml

def log_resources(label, init_mem=None, init_cpu=None):
    process = psutil.Process()
    current_mem = process.memory_info().rss / (1024**2)  # Convert bytes to MB
    current_cpu = psutil.cpu_percent(interval=2)  # 2-second interval for stable reading
    if init_mem is not None and init_cpu is not None:
        mem_delta = current_mem - init_mem
        cpu_delta = current_cpu - init_cpu
        print(f"{label} - Memory Consumed: {mem_delta:.2f} MB, CPU Usage Delta: {cpu_delta}%")
    else:
        print(f"{label} - Memory Usage: {current_mem:.2f} MB, CPU Usage: {current_cpu}%")
    return current_mem, current_cpu

def log_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Utilization: {gpu_util.gpu}% | Memory Used: {mem_info.used / (1024 ** 2):.2f} MB")
    pynvml.nvmlShutdown()

def log_slice_properties(img, multiplier, scale_factor):
    data = img.get_fdata()
    shape = data.shape
    affine = img.affine
    voxel_size = nib.affines.voxel_sizes(affine)
    expanded_slices = shape[2] * multiplier
    scaled_shape = (int(shape[0] * scale_factor), int(shape[1] * scale_factor), expanded_slices)
    estimated_size = (data.nbytes * multiplier * scale_factor**2) / (1024**2)

    print(f"Original Dimensions (x, y, z): {shape}")
    print(f"Voxel size (mm): {voxel_size}")
    print(f"Total slices after expansion: {expanded_slices}")
    print(f"Scaled Dimensions (x, y): {scaled_shape[:2]}")
    print(f"Data type: {data.dtype}")
    print(f"Estimated expanded file size: {estimated_size:.2f} MB")

def expand_slices(data, multiplier, array_module):
    return array_module.repeat(data, multiplier, axis=2)

def resize_slices(data, scale_factor, array_module):
    return zoom(data, (scale_factor, scale_factor, 1), order=3)

def process_t1_image(input_path, output_path, multiplier=1, scale_factor=1, use_gpu=False):
    total_start = time.time()  # Start time
    array_module = cp if use_gpu else np
    total_start = time.time()
    init_mem, init_cpu = log_resources("Before Start")

    phases = {}
    resource_deltas = []

    # Phase 1: Load NIfTI Data
    start = time.time()
    img = nib.load(input_path)
    data = array_module.asarray(img.get_fdata())  # Transfer to GPU if GPU is used
    end = time.time()
    mem_delta, cpu_delta = log_resources("After Load NIfTI", init_mem, init_cpu)
    phases['Load Time'] = end - start
    resource_deltas.append((mem_delta, cpu_delta, phases['Load Time']))

    # Log slice properties
    log_slice_properties(img, multiplier, scale_factor)
    

    # Resize slices if scale_factor > 1
    if scale_factor > 1:
        data = resize_slices(data, scale_factor, array_module)
    mem_delta, cpu_delta = log_resources("After Resize", init_mem, init_cpu)
    resource_deltas.append((mem_delta, cpu_delta, phases['Load Time']))

    # Expand slices if multiplier > 1
    if multiplier > 1:
        data = expand_slices(data, multiplier, array_module)
    log_gpu_utilization()
    mem_delta, cpu_delta = log_resources("After Expand Slices", init_mem, init_cpu)
    resource_deltas.append((mem_delta, cpu_delta, phases['Load Time']))

    # Phase 2: Preprocessing
    start = time.time()
    data = array_module.clip(data, array_module.percentile(data, 5), array_module.percentile(data, 95))
    data = (data - data.min()) / (data.max() - data.min())
    end = time.time()
    log_gpu_utilization()
    mem_delta, cpu_delta = log_resources("After Preprocessing", init_mem, init_cpu)
    phases['Preprocessing Time'] = end - start
    resource_deltas.append((mem_delta, cpu_delta, phases['Preprocessing Time']))

    # Phase 3: Transformation
    start = time.time()
    processed_data = ndimage.gaussian_filter(data.get() if use_gpu else data, sigma=1)
    processed_data = array_module.asarray(processed_data) if use_gpu else processed_data
    end = time.time()
    log_gpu_utilization()
    mem_delta, cpu_delta = log_resources("After Transformation", init_mem, init_cpu)
    phases['Transformation Time'] = end - start
    resource_deltas.append((mem_delta, cpu_delta, phases['Transformation Time']))

    # Phase 4: Save Processed Image
    start = time.time()
    expanded_img = nib.Nifti1Image(processed_data.get() if use_gpu else processed_data, img.affine, img.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(expanded_img, output_path)
    end = time.time()
    log_gpu_utilization()
    mem_delta, cpu_delta = log_resources("After Save", init_mem, init_cpu)
    phases['Save Time'] = end - start
    resource_deltas.append((mem_delta, cpu_delta, phases['Save Time']))

    # Calculate Weighted Averages for Memory and CPU Usage
    total_phase_time = sum(duration for _, _, duration in resource_deltas)
    weighted_avg_memory = sum(mem * (duration / total_phase_time) for mem, _, duration in resource_deltas)
    weighted_avg_cpu = sum(cpu * (duration / total_phase_time) for _, cpu, duration in resource_deltas)

    # Print Weighted Averages and Summary
    print(f"\nWeighted Average Memory Usage: {weighted_avg_memory:.2f} MB")
    print(f"Weighted Average CPU Usage: {weighted_avg_cpu:.2f}%")

    # Summary of timings
    print("\nSummary of Timings:")
    for phase, duration in phases.items():
        print(f"{phase}: {duration:.4f} seconds")
    total_end = time.time()  # End time
    total_time = total_end - total_start
    print(f"\nTotal Execution Time: {total_time:.4f} seconds")


# Run the function using command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input NIfTI file")
    parser.add_argument("output_path", help="Path to save processed NIfTI file")
    parser.add_argument("--multiplier", type=int, default=1, help="Slice expansion multiplier")
    parser.add_argument("--scale_factor", type=float, default=1, help="Slice resizing scale factor")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")

    args = parser.parse_args()

    process_t1_image(args.input_path, args.output_path, args.multiplier, args.scale_factor, args.gpu)
