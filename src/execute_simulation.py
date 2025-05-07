from multiprocessing import Pool, cpu_count

import numpy as np
from constants import *
from general_radon_transform import generalised_radon_transform
from x_ray_spect import generate_source_spectrum, source_spectrum_effective_energy

debug: bool = False

source_spect = generate_source_spectrum()
source_effective_energy = source_spectrum_effective_energy(source_spect)

# For sinogram generation, we'd like to go column by column (slice by slice)
def generalised_radon_transform_col(phantom: np.ndarray,
                                    theta: float,
                                    detec_col: int,
                                    source_spect: dict[float, float],
                                    bin_linspace: np.ndarray) -> list[list[float]]:
  detec_col_output: list[list[float]] = []
  for detec_row in range(0, detec_size):
    detec_col_output.append(generalised_radon_transform(phantom, theta, (detec_row, detec_col), source_spect, bin_linspace))
  return detec_col_output


def radon_over_detector_pixels(phantom: np.ndarray, detec_size: int, detec_dims: int, detec_bins: int, sinogram_slice: int | None) -> np.ndarray:
  sinogram_mode: bool = True if sinogram_slice is not None else False
  
  detec_shape: list | tuple = []

  if not sinogram_mode:
    for i in range(0, detec_dims):
      detec_shape.append(detec_size)
  else:
    detec_shape.append(detec_size)
    detec_shape.append(len(theta_linspace)) # Round to nearest 100

  detec_shape.append(detec_bins) # Our bins are not included in the dims size, but we tag it on!
  detec_shape = tuple(detec_shape)
  detec_output: np.ndarray = np.zeros(detec_shape)

  if debug:
    print("DEBUG MODE: Running in serial mode")
    max_rows: int = 1
    for detec_row in range(0, max_rows):
      print(f"Processing row {detec_row}")
      detec_output[detec_row,:] = generalised_radon_transform_col(phantom, theta, detec_row, source_spect, bin_linspace)
    return detec_output

  # Trying out multiprocessing
  cpus: int = cpu_count()
  with Pool(processes=cpus) as pool:

    reses: dict[tuple, np.ndarray] = {}
    
    if not sinogram_mode:
      for y in range(0, detec_size):
        res = pool.apply_async(generalised_radon_transform_col, args=(phantom,
                                                                      theta,
                                                                      y,
                                                                      source_spect,
                                                                      bin_linspace
                                                                      ))
        reses[(y)] = res
      for y in range(0, detec_size):
        detec_output[:,y] = reses[(y)].get()
    else:
      for sinogram_theta in theta_linspace.tolist():
        res = pool.apply_async(generalised_radon_transform_col, args=(phantom,
                                                                      sinogram_theta,
                                                                      sinogram_slice,
                                                                      source_spect,
                                                                      bin_linspace
                                                                      ))
        reses[(sinogram_theta)] = res
      for sinogram_index, sinogram_theta in enumerate(theta_linspace):
        detec_output[:,sinogram_index] = reses[(sinogram_theta)].get()

  return detec_output