import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_func

from constants import *


# --- Kramers Law Spectrum ---
def kramers_spectrum(E, E0=E0):
    """
    Kramers law: I(E) ∝ (E0 - E)/E for 0 < E < E0.
    For E=0, we define I=0 to avoid division by zero.
    """
    I = np.zeros_like(E)
    valid = (E > 0) & (E < E0)
    I[valid] = (E0 - E[valid]) / E[valid]
    return I

# --- Beta Distribution–Type Spectrum ---
def beta_spectrum(E, E0=E0, alpha=2.0, beta_param=2.0):
    """
    I(E) = I0 * (E/E0)^(alpha-1) * (1 - E/E0)^(beta_param-1)
    for 0 <= E <= E0. The function is normalized (I0 set so that
    the area under the curve equals 1) using the beta function.
    """
    x = E / E0
    I = np.zeros_like(E)
    valid = (E >= 0) & (E <= E0)
    I[valid] = x[valid]**(alpha-1) * (1 - x[valid])**(beta_param-1)
    # Normalize so that the integral over [0, E0] equals 1:
    norm = beta_func(alpha, beta_param) * E0
    return I / norm

# --- Define Energy Grid ---
# We avoid E=0 for Kramers law to prevent division by zero.
E = np.linspace(0.001, E0, 1000)

# Calculate spectra
I_kramers = kramers_spectrum(E, E0)
I_beta = beta_spectrum(E, E0, alpha=2.0, beta_param=6.0)

# Normalize for plotting (maximum value = 1)
I_kramers_norm = I_kramers / np.max(I_kramers)
I_beta_norm = I_beta / np.max(I_beta)

# --- Plotting ---
'''
plt.figure(figsize=(8,6))
plt.plot(E, I_kramers_norm, label="Kramers Spectrum (normalized)")
plt.plot(E, I_beta_norm, label="Beta Distribution Spectrum (normalized)")
plt.xlabel("Photon Energy E (MeV)")
plt.ylabel("Normalized Intensity")
plt.title(f"{E0} MeV X-ray Source Spectrum")
plt.legend()
plt.grid(True)
plt.show()
'''

def generate_source_spectrum_beta() -> dict[float, float]: # Key: E, Value: Intensity
    source_linspace: np.ndarray = np.linspace(detec_min_e_mev, E0, detec_bins)
    spect = beta_spectrum(source_linspace, E0, alpha=2.0, beta_param=4.0)
    spect: dict[float, float] = dict(zip(source_linspace, spect))
    return spect

# Abstract this into one function that just serves up the spectrum that we'll use for our simulations
def generate_source_spectrum() -> dict[float, float]: # Key: E, Value: Intensity
    spect = {E0:1.0} # Monochromatic. E0 has intensity 1.0
    return spect

def source_spectrum_effective_energy(source_spectrum: dict[float, float]) -> float:
    # Compute a weighted average of the spectrum
    return np.average(list(source_spectrum.keys()), weights=list(source_spectrum.values()))
