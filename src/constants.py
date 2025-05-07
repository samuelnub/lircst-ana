
import numpy as np

# ---------------------
# Constants
# ---------------------
r0 = 2.8179403227e-15  # Classical electron radius in meters
m_e_c2 = 0.511         # Electron rest energy in MeV

phan_size: int = 128
phan_channels: int = 2 # Channel 0: Scattering Coefficient, Channel 1: Attenuation Coefficient
chan_scat: int = 0
chan_atten: int = 1

E0 = 1.0 # Effective "peak" energy of a 6MeV radiotherapy beam
voxel_physical_scale_len: float = 0.001 # 1 metre to 1mm thereabouts
voxel_physical_scale_atten: float = 0.01 # For our mass attenuation coefficients, which use cm^2. We will convert to mm^2
theta: float = np.pi * (-2/6) # Unused in sinogram mode
detec_size: int = phan_size
detec_dims: int = 2
detec_bins: int = 100
detec_max_e_mev: float = E0
detec_min_e_mev: float = 0.001
detec_dist_isocenter: int = 0.5 * (1/voxel_physical_scale_len) # 0.5m from iso
source_dist_isocenter: int = 1 * (1/voxel_physical_scale_len) # 1m from iso

bin_linspace: np.ndarray = np.linspace(detec_min_e_mev, detec_max_e_mev, detec_bins) # These two linspaces may well be the same
theta_linspace: np.ndarray = np.linspace(0, 1*np.pi, int(round(np.pi * detec_size / 2, -2)), endpoint=False) # Only image 1 pi (180 degrees)

#source_spect: dict[float, float] | None = None # Defined in a little bit
#source_effective_energy: float | None = None # Defined in a little bit