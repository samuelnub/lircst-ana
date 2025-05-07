import numpy as np

from constants import *
from path_tracer import siddon_jacobs_3d
from scatter_probability import klein_nishina_dsigma_domega
from grid_detec_response import grid_response, detec_response


def generalised_radon_transform(phantom: np.ndarray,
                                theta: float,
                                detec_point: tuple,
                                source_spect: dict[float, float], # Key: Energy, Value: Intensity
                                bin_linspace: np.ndarray) -> list[float]: # Energy bins for this pixel
  # Take line integral for our given theta and detec_point (coordinates on the detector grid, final axis is energy bin)

  e_intensities: dict[float, float] = {} # Key: Energy, Value: Accumulated intensity

  # Small nested function to check if a given coord is within the phantom bounds
  def is_within_bounds(coord: tuple) -> bool:
    return 0 <= coord[0] < phantom.shape[-3] and 0 <= coord[1] < phantom.shape[-2] and 0 <= coord[2] < phantom.shape[-1]

  # We'll rotate around the z axis
  detec_point_physical: np.ndarray = np.array([-detec_dist_isocenter, detec_point[0], detec_point[1]]) # left side
  detec_point_physical_opposite: np.ndarray = np.array([detec_dist_isocenter, detec_point[0], detec_point[1]]) # left side
  source_point_physical: np.ndarray = np.array([0, source_dist_isocenter, 0]) # above

  # Move all points so that our phantom centre is the new origin, before we rotate
  detec_point_physical = detec_point_physical - phan_size * 0.5
  detec_point_physical_opposite = detec_point_physical_opposite - phan_size * 0.5
  source_point_physical = source_point_physical - phan_size * 0.5

  # Translate all points around the z axis by theta
  rot_mat: np.ndarray = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
  detec_point_physical = rot_mat @ detec_point_physical.T
  detec_point_physical_opposite = rot_mat @ detec_point_physical_opposite.T
  source_point_physical = rot_mat @ source_point_physical.T

  # Move all points back to their original positions
  detec_point_physical = detec_point_physical + phan_size * 0.5
  detec_point_physical_opposite = detec_point_physical_opposite + phan_size * 0.5
  source_point_physical = source_point_physical + phan_size * 0.5

  # Let's just turn our points into tuples for ease
  detec_point_physical = (detec_point_physical[0], detec_point_physical[1], detec_point_physical[2])
  detec_point_physical_opposite = (detec_point_physical_opposite[0], detec_point_physical_opposite[1], detec_point_physical_opposite[2])
  source_point_physical = (source_point_physical[0], source_point_physical[1], source_point_physical[2])

  # Get traversed pixels from detec_point_physical to detec_point_physical_opposite
  rd_r_voxels_all, rd_r_voxels_all_lens = siddon_jacobs_3d(detec_point_physical, detec_point_physical_opposite, phan_size)

  # Every voxel in this line is the result of a scatter incident, tracing back to the source.
  # So for each of these voxels (r's), also get the siddon result for r back to our source

  # We can save (very little) computation time by having a running tally of our rd_r attenuation integral
  # This is under the assumption that our iterations go from detector -> point r
  atten_rd_r_exp_running_integral: float = 0

  for _, (rd_r_voxel, rd_r_voxel_len) in enumerate(dict(zip(rd_r_voxels_all, rd_r_voxels_all_lens)).items()):

    if not is_within_bounds(rd_r_voxel):
      continue

    # Add to the rd_r attenuation tally before we keep looping over E's
    atten_rd_r_exp_running_integral = atten_rd_r_exp_running_integral + phantom[chan_atten][rd_r_voxel] * rd_r_voxel_len * voxel_physical_scale_atten

    # Grab our scattering coefficient now, and if no scattering can occur here,
    # we do not even need to consider propagating back up to r_rs
    r_scat_coef: float = phantom[chan_scat][rd_r_voxel]
    if r_scat_coef == 0:
      continue

    # Get phi between this voxel and source
    r_rd_vec: tuple = (detec_point_physical[0] - rd_r_voxel[0], detec_point_physical[1] - rd_r_voxel[1], detec_point_physical[2] - rd_r_voxel[2])
    r_rs_vec: tuple = (source_point_physical[0] - rd_r_voxel[0], source_point_physical[1] - rd_r_voxel[1], source_point_physical[2] - rd_r_voxel[2])
    phi: float = np.pi - np.arccos(np.dot(r_rd_vec, r_rs_vec) / (np.linalg.norm(r_rd_vec) * np.linalg.norm(r_rs_vec)))

    # Get all the voxels between this voxel and the source
    r_rs_voxels, r_rs_voxels_lens = siddon_jacobs_3d(rd_r_voxel, source_point_physical, phan_size)

    atten_r_rs_exp_integral: float = 0
    for r_rs_voxel, r_rs_voxel_len in zip(r_rs_voxels, r_rs_voxels_lens):
      if not is_within_bounds(r_rs_voxel):
        continue
      atten_r_rs_exp_integral = atten_r_rs_exp_integral + phantom[chan_atten][r_rs_voxel] * r_rs_voxel_len * voxel_physical_scale_atten

    # We can now compute the inverse square laws for this voxel from source and detector
    # This is under the assumption that we have a cone-beam source - a hypothetical perfect parallel beam would not need this
    inv_sq_rd_r: float = 1 / (np.abs(np.linalg.norm(np.subtract(rd_r_voxel, detec_point_physical))) * voxel_physical_scale_len) # L2 norm is Euclid dist
    inv_sq_r_rs: float = 1 / (np.abs(np.linalg.norm(np.subtract(rd_r_voxel, source_point_physical))) * voxel_physical_scale_len) # L2 norm is Euclid dist

    # Let's cache computed Klein-Nishina values for each E', as many will be reused (when phi is fixed)
    klein_nishina_cache_Eprime_fixed_phi: dict[float, float] = {}

    # Let's integrate over all our desired energy bins
    for _, E in np.ndenumerate(bin_linspace):
      e_intensity: float | None = None
      
      # Placeholders
      grid_detec_response = grid_response() * detec_response()

      # Let's integrate over E' which will be from E up to E0, discretised
      # Nested loop: For each voxel from r to rs, see how the intensity lowers due to atten. + inverse sq. law.
      # These are all the energies that RESULT in our E, so sum up these intensities as they come
      eprimes_intensity_integral: float = 0
      for _, Eprime in enumerate(source_spect): # Keys are Eprimes
        # A quick check here to see if our E' is greater than E
        if Eprime < E:
          continue

        eprime_intensity: float | None = None

        source_I_at_Eprime: float = source_spect[Eprime]
        scaled_I_r_rs: float = source_I_at_Eprime * inv_sq_r_rs

        # At the end of that voxel loop, we can then compute the Klein-Nishina prob that our scat originating from E' happens
        # and scale our tally accordingly
        klein_nishina_prob: float | None = None
        if Eprime in klein_nishina_cache_Eprime_fixed_phi:
          klein_nishina_prob = klein_nishina_cache_Eprime_fixed_phi[Eprime]
        else:
          klein_nishina_prob = klein_nishina_dsigma_domega(phi, Eprime)
          klein_nishina_cache_Eprime_fixed_phi[Eprime] = klein_nishina_prob

        eprime_intensity = scaled_I_r_rs * (np.e**(-atten_r_rs_exp_integral)) * klein_nishina_prob
        eprimes_intensity_integral = eprimes_intensity_integral + eprime_intensity

      # Now we can compute the final intensity from scatter point r from source all the way to detector, for this energy bin
      e_intensity = grid_detec_response * (np.e**(-atten_rd_r_exp_running_integral)) * inv_sq_rd_r * eprimes_intensity_integral
      # Don't forget to tag on our scattering coefficient!
      # Teeechnically this should be applied to the whole integral if our logic is for total energy integration
      # But for us, we are discretising bins
      e_intensity = e_intensity * r_scat_coef

      if E in e_intensities:
        e_intensities[E] = e_intensities[E] + e_intensity
      else:
        e_intensities[E] = e_intensity

  detec_point_output: list[float] = []
  for _, bin_E in np.ndenumerate(bin_linspace):
    detec_point_output.append(e_intensities[bin_E] if bin_E in e_intensities else 0)
  return detec_point_output