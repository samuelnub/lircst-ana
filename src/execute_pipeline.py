import random
import os
import time

import numpy as np
from constants import *
from phantom_generation import create_random_voxel_phantom, create_blank_voxel_phantom
from execute_simulation import radon_over_detector_pixels
from x_ray_spect import generate_source_spectrum, source_spectrum_effective_energy

source_spect = generate_source_spectrum()
source_effective_energy = source_spectrum_effective_energy(source_spect)

def generate_data_pipeline():
    # Generate N amount of phantoms
    # For each one, we take sinogram slices as individual samples
    # We will rely on the default configuration for our parameters

    N: int = 125
    for i in range(0, N):
        phantom, phan_metadata = create_random_voxel_phantom()

        print(f"Generated phantom {i+1}/{N} with metadata: {phan_metadata}")

        # Create timestamp-based file directory for this phantom
        timestamp: float = time.time()
        file_dir: str = f"../data/{int(timestamp)}"
        os.makedirs(file_dir)

        metadata: dict = {
            "voxel_physical_scale_len": voxel_physical_scale_len,
            "voxel_physical_scale_atten": voxel_physical_scale_atten,
            "source_dist_isocenter": source_dist_isocenter,
            "source_spect": source_spect,
            "bin_linspace": bin_linspace,
            "theta_linspace": theta_linspace,
            "phan_metadata": phan_metadata,
        }

        # Save the metadata
        np.save(f"{file_dir}/meta.npy", metadata)

        # To check whether our slices are "empty", let's first get the scat and atten coef of our background material
        phan_background_name: str = phan_metadata["background_material"]
        blank_phantom = create_blank_voxel_phantom(phan_background_name, phantom.shape)

        slice_indices: list[int] = list(range(0, detec_size))
        random.shuffle(slice_indices) # Reduce bias in our data
        for j, slice_index in enumerate(slice_indices):
            # Generate ground-truth phantom slice first
            phan: np.ndarray = phantom[:, :, :, slice_index]
            blank_phan: np.ndarray = blank_phantom[:, :, :, slice_index]
            # Check if the slice is empty
            if not np.subtract(phan, blank_phan).any():
                # Empty slice, so we skip it
                continue
            np.save(f"{file_dir}/phan-{slice_index}.npy", phan)

            timestamp_delta_a: float = time.time()

            sinogram_output: np.ndarray = radon_over_detector_pixels(phantom, 
                                                                     detec_size, 
                                                                     detec_dims, 
                                                                     detec_bins, 
                                                                     sinogram_slice=slice_index)
            # Save the sinogram output
            np.save(f"{file_dir}/sino-{slice_index}.npy", sinogram_output)

            timestamp_delta_b: float = time.time()
            print(f"Generated sinogram slice {slice_index} ({j+1}/{len(slice_indices)}) for phantom {int(timestamp)} ({i}) in {round(timestamp_delta_b - timestamp_delta_a)} seconds")


generate_data_pipeline()