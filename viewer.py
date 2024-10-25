import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import time
import numpy as np

def render_3d(nifti_path):
    # Load the NIfTI file and render a 3D view
    img = nib.load(nifti_path)
    plotting.view_img(img, threshold=None).open_in_browser()

def view_slice(nifti_path, slice_number):
    # Load the NIfTI file and display a specific slice
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    print_slice_properties(img)
    
    plt.imshow(data[:, :, slice_number], cmap='gray')
    plt.title(f"Slice {slice_number}")
    plt.axis('off')
    plt.show()

def roll_slices(nifti_path, delay=1):
    # Load the NIfTI file and display slices one by one with a delay
    img = nib.load(nifti_path)
    data = img.get_fdata()
    num_slices = data.shape[2]

    print_slice_properties(img)
    
    for i in range(num_slices):
        plt.imshow(data[:, :, i], cmap='gray')
        plt.title(f"Slice {i}")
        plt.axis('off')
        plt.pause(delay)
        plt.clf()  # Clear the figure to prepare for the next slice

def print_slice_properties(img):
    data = img.get_fdata()
    shape = data.shape
    affine = img.affine
    voxel_size = nib.affines.voxel_sizes(affine)
    
    print(f"Dimensions (x, y, z): {shape}")
    print(f"Voxel size (mm): {voxel_size}")
    print(f"Total number of slices: {shape[2]}")
    print(f"Data type: {data.dtype}")
    print(f"File size: {data.nbytes / (1024**2):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="MRI Viewer for NIfTI files")
    parser.add_argument("nifti_path", help="Path to the NIfTI file (.nii or .nii.gz)")
    parser.add_argument("mode", choices=["render", "slice", "roll"], help="Display mode: render (3D view), slice (specific slice), or roll (slide show)")
    parser.add_argument("--slice_number", type=int, help="Slice number for 'slice' mode")
    parser.add_argument("--delay", type=float, default=1, help="Delay in seconds for 'roll' mode")

    args = parser.parse_args()

    if args.mode == "render":
        render_3d(args.nifti_path)
    elif args.mode == "slice":
        if args.slice_number is None:
            print("Error: Please provide --slice_number for slice mode")
        else:
            view_slice(args.nifti_path, args.slice_number)
    elif args.mode == "roll":
        roll_slices(args.nifti_path, args.delay)

if __name__ == "__main__":
    main()
