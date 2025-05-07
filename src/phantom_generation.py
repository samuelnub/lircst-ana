# Voxelised 3D medical imaging phantom (comprising of random shapes) in numpy ndarray format

import numpy as np

from constants import *
from material_loader import MaterialLoader
from x_ray_spect import generate_source_spectrum, source_spectrum_effective_energy

source_spect = generate_source_spectrum()
source_effective_energy = source_spectrum_effective_energy(source_spect)

material_loader = MaterialLoader(prop_data_path="/home/samnub/dev/lircst-ana/res/nist_ele_prop.json", 
                                 atten_data_path="/home/samnub/dev/lircst-ana/res/nist_ele_atten.json",
                                 prop_comp_data_path="/home/samnub/dev/lircst-ana/res/nist_comp_prop.json",
                                 atten_comp_data_path="/home/samnub/dev/lircst-ana/res/nist_comp_atten.json",)


def create_phantom_cylinder_mask(shape=(phan_channels, phan_size, phan_size, phan_size)) -> np.ndarray:
  # Create a cylindrical mask for the phantom
  # The cylinder is centered in the phantom and has a radius of phan_size/2
  # The cylinder is oriented along the z-axis

  # Generate voxel indices for the shape
  x_indices = np.arange(shape[-3])
  y_indices = np.arange(shape[-2])
  z_indices = np.arange(shape[-1])
  x, y, z = np.meshgrid(x_indices, y_indices, z_indices)

  mask = np.sqrt((x - phan_size//2)**2 + (y - phan_size//2)**2) # z is our axis of rotation, so ignore it
  return mask < phan_size//2


def create_blank_voxel_phantom(background_material: str, shape=(phan_channels, phan_size, phan_size, phan_size)) -> np.ndarray:
    phantom = np.zeros(shape, dtype=np.float32)
    # Set background material
    # Generate voxel indices for the shape
    x_indices = np.arange(shape[-3])
    y_indices = np.arange(shape[-2])
    z_indices = np.arange(shape[-1])
    x, y, z = np.meshgrid(x_indices, y_indices, z_indices)
    # Cylindrical phantom
    voxel_indices = x + y + z
    # Remove all voxels outside cylinder radius phan_size/2 from centre of phantom
    mask = create_phantom_cylinder_mask(shape)
    voxel_indices = np.logical_and(voxel_indices, mask)
    phantom[chan_scat][voxel_indices] = material_loader.calc_electron_density(background_material)
    phantom[chan_atten][voxel_indices] = material_loader.calc_mass_atten_coeff_at_energy(background_material, source_effective_energy)

    return phantom


def create_random_voxel_phantom(shape=(phan_channels, phan_size, phan_size, phan_size)) -> tuple[np.ndarray, dict]:
    """
    Creates a voxelized 3D medical imaging phantom with random shapes.

    Args:
      shape: The shape of the 3D array (c, x, y, z).
      num_shapes: The number of random shapes to generate.

    Returns:
      A NumPy ndarray representing the 3D phantom.
    """

    metadata: dict = {
        "shape": shape,
        "inserts": [],
    }

    mu = phan_size // 2
    sigma = phan_size // np.random.randint(4, 12) # Standard deviation for Gaussian distribution

    phantom: np.ndarray | None = None

    # We'll have three types of phantom: "Medical" and "Industrial" and "Random" - Fleshy vs. Inorganic (mostly) vs Random

    background_materials: list[str] = []
    insert_materials: list[str] = []

    phan_type = np.random.choice(["medical", "industrial", "random"])

    if phan_type == "medical":
      background_materials = ["Adipose Tissue (ICRU-44)", 
                             "Muscle, Skeletal (ICRU-44)", 
                             "Lung Tissue (ICRU-44)",
                             "Tissue, Soft (ICRU-44)",
                             "Tissue, Soft (ICRU Four-Component)",
                             "Nothing", # Faster to compute than Air
                             "Water, Liquid"]
      insert_materials = ["Bone, Cortical (ICRU-44)",
                         "Blood, Whole (ICRU-44)",
                         "Adipose Tissue (ICRU-44)",
                         "Muscle, Skeletal (ICRU-44)",
                         "Air, Dry (near sea level)",
                         "Brain, Grey/White Matter (ICRU-44)",
                         "Ovary (ICRU-44)",
                         "Testis (ICRU-44)",
                         "Eye Lens (ICRU-44)",
                         "Water, Liquid"] # This is a horrific amalgamation of human body parts
    elif phan_type == "industrial":
      background_materials = ["Nothing", # Faster to compute than Air
                             "Water, Liquid"]
      insert_materials = ["Glass, Lead",
                         "Lithium Fluride",
                         "Lithium Tetraborate",
                         "Bakelite",
                         "Magnesium Tetroborate",
                         "Mercuric Iodide",
                         "Photographic Emulsion (Kodak Type AA)",
                         "Photographic Emulsion  (Standard Nuclear)",
                         "C-552 Air-equivalent Plastic",
                         "Plastic Scintillator, Vinyltoluene",
                         "Cadmium Telluride",
                         "Polyethylene",
                         "Calcium Fluoride",
                         "Polyethylene Terephthalate, (Mylar)",
                         "Calcium Sulfate",
                         "Polymethyl Methacrylate",
                         "15 mmol L-1 Ceric Ammonium Sulfate Solution",
                         "Polystyrene",
                         "Cesium Iodide",
                         "Polytetrafluoroethylene, (Teflon)",
                         "Concrete, Ordinary",
                         "Polyvinyl Chloride",
                         "Concrete, Barite (TYPE BA)",
                         "Radiochromic Dye Film, Nylon Base",
                         "Ferrous Sulfate Standard Fricke",
                         "Gadolinium Oxysulfide",
                         "Gafchromic Sensor",
                         "Gallium Arsenide",
                         "Glass, Borosilicate (Pyrex)",
                         "Water, Liquid",
                         "Air, Dry (near sea level)",
                         "1", "2", "3", "6", "7", "8", "10", "11", "12", "13", "14", "17", "20", "22", "24", "26", "27", "28", "29", "30", "35", "38", "43", "47", "50", "53", "60", "74", "79", "80", "82", "92"] # Some atomic numbers for pure elements
    elif phan_type == "random":
      background_materials = ["Nothing"]
      insert_materials = material_loader.get_all_material_names()

    min_shapes: int = 1
    max_shapes: int | None = None
    if phan_type == "medical":
      max_shapes = 8
    elif phan_type == "industrial":
      max_shapes = 16
    elif phan_type == "random":
      max_shapes = 24
    num_shapes: int = np.random.randint(min_shapes, max_shapes + 1)  # Random number of shapes

    anomaly_chance: float | None = None
    if phan_type == "medical":
      anomaly_chance = 0.4
    elif phan_type == "industrial":
      anomaly_chance = 0.1
    elif phan_type == "random":
      anomaly_chance = 0.25

    anomaly_min_scale: float | None = None
    anomaly_max_scale: float | None = None
    if phan_type == "medical":
      anomaly_min_scale = 0.8
      anomaly_max_scale = 4.0 # Dense tumours for example
    elif phan_type == "industrial":
      anomaly_min_scale = 0.9
      anomaly_max_scale = 1.1
    elif phan_type == "random":
      anomaly_min_scale = 0.5
      anomaly_max_scale = 2.0

    # Randomly selected background material
    background_material = np.random.choice(background_materials)

    phantom = create_blank_voxel_phantom(background_material, shape)

    metadata["background_material"] = background_material
    metadata["type"] = phan_type

    for _ in range(num_shapes):
      # Randomly determine the shape of each object
      # Let's say we want more spheres and ellipsoids than cubes
      shape_type = np.random.choice(["sphere", "cube", "cuboid", "ellipsoid"])

      # Random gaussian centre for the shape (centred around mid-phantom)
      center_x = int(np.random.normal(mu, sigma))
      center_y = int(np.random.normal(mu, sigma))
      center_z = np.random.randint(0, shape[-1]) # Uniform distribution for z-axis as it's our axis of rotation

      radius = np.random.randint(min(shape[-3:]) // 32, min(shape[-3:]) // 4) # Adjust radius based on phantom size

      # Generate voxel indices for the shape
      x_indices = np.arange(shape[-3])
      y_indices = np.arange(shape[-2])
      z_indices = np.arange(shape[-1])
      x, y, z = np.meshgrid(x_indices, y_indices, z_indices)

      if shape_type == "sphere":
        # Draw a sphere
        voxel_indices = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
      elif shape_type == "cube":
        # Draw a cube
        voxel_indices = np.logical_and(np.abs(x - center_x) <= radius,
                                        np.logical_and(np.abs(y - center_y) <= radius,
                                                        np.abs(z - center_z) <= radius))
      elif shape_type == "cuboid":
        # Draw a cuboid
        # Randomly select dimensions for the cuboid
        length = radius * np.random.uniform(0.1, 2)
        width = radius * np.random.uniform(0.1, 2)
        height = radius * np.random.uniform(0.1, 2)
        voxel_indices = np.logical_and(np.abs(x - center_x) <= length,
                                        np.logical_and(np.abs(y - center_y) <= width,
                                                        np.abs(z - center_z) <= height))
      elif shape_type == "ellipsoid":
        # Draw an ellipsoid with different radii along each axis
        a = radius * np.random.uniform(0.1, 2) # x-axis radius
        b = radius * np.random.uniform(0.1, 2) # y-axis radius
        c = radius * np.random.uniform(0.1, 2) # z-axis radius
        voxel_indices = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2) + ((z - center_z)**2 / c**2) <= 1
      else:
        continue

      # Update the phantom
      # Randomly selected insert material
      insert_material = np.random.choice(insert_materials)
      scat_val = material_loader.calc_electron_density(insert_material)
      atten_val = material_loader.calc_mass_atten_coeff_at_energy(insert_material, source_effective_energy)
      # if anomoly, vary the material properties
      is_anomaly = False
      if np.random.random() < anomaly_chance:
        is_anomaly = True
        anomaly_scale = np.random.uniform(anomaly_min_scale, anomaly_max_scale)
        scat_val *= anomaly_scale * np.random.uniform(0.95, 1.05)
        atten_val *= anomaly_scale * np.random.uniform(0.95, 1.05)
      

      # Apply a 3D rotation matrix to the voxel indices
      rand_theta_x = np.random.uniform(0, 2*np.pi)
      rand_theta_y = np.random.uniform(0, 2*np.pi)
      rand_theta_z = np.random.uniform(0, 2*np.pi)
      # Rotation matrices
      Rx = np.array([[1, 0, 0],
                     [0, np.cos(rand_theta_x), -np.sin(rand_theta_x)],
                     [0, np.sin(rand_theta_x), np.cos(rand_theta_x)]])
      Ry = np.array([[np.cos(rand_theta_y), 0, np.sin(rand_theta_y)],
                     [0, 1, 0],
                     [-np.sin(rand_theta_y), 0, np.cos(rand_theta_y)]])
      Rz = np.array([[np.cos(rand_theta_z), -np.sin(rand_theta_z), 0],
                      [np.sin(rand_theta_z), np.cos(rand_theta_z), 0],
                      [0, 0, 1]])
      
      # Let's supersample our voxel points to get a smoother rotation
      scale_factor = 2

      x_super_indices = np.arange(shape[-3] * scale_factor)
      y_super_indices = np.arange(shape[-2] * scale_factor)
      z_super_indices = np.arange(shape[-1] * scale_factor)
      x_super, y_super, z_super = np.meshgrid(x_super_indices, y_super_indices, z_super_indices)

      voxel_indices_super = np.repeat(voxel_indices, scale_factor, axis=0)
      voxel_indices_super = np.repeat(voxel_indices_super, scale_factor, axis=1)
      voxel_indices_super = np.repeat(voxel_indices_super, scale_factor, axis=2)

      coords_super = np.array([x_super[voxel_indices_super], y_super[voxel_indices_super], z_super[voxel_indices_super]])

      # Translate point of rotation to center_x, center_y, center_z
      coords_super[0] -= center_x*scale_factor
      coords_super[1] -= center_y*scale_factor
      coords_super[2] -= center_z*scale_factor

      # Apply the rotation
      coords_rotated_super = np.dot(Rz, np.dot(Ry, np.dot(Rx, coords_super)))
      x_rotated_super = coords_rotated_super[0].astype(int)
      y_rotated_super = coords_rotated_super[1].astype(int)
      z_rotated_super = coords_rotated_super[2].astype(int)

      # Translate back to original coordinates
      x_rotated_super += center_x*scale_factor
      y_rotated_super += center_y*scale_factor
      z_rotated_super += center_z*scale_factor

      # Ensure the rotated coordinates are within bounds
      x_rotated_super = np.clip(x_rotated_super, 0, (phan_size-1)*scale_factor)
      y_rotated_super = np.clip(y_rotated_super, 0, (phan_size-1)*scale_factor)
      z_rotated_super = np.clip(z_rotated_super, 0, (phan_size-1)*scale_factor)

      # Create a new mask for the rotated coordinates
      rotated_voxel_indices_super = np.zeros_like(voxel_indices_super)
      rotated_voxel_indices_super[voxel_indices_super] = False
      rotated_voxel_indices_super[x_rotated_super, y_rotated_super, z_rotated_super] = True
      voxel_indices_super = rotated_voxel_indices_super

      # Downsample the supersampled voxel indices to the original size (ensuring True values are kept)
      voxel_indices = np.zeros_like(voxel_indices)

      # Iterate over all supersampled voxels
      for i in range(0, phan_size*scale_factor, 1):
        for j in range(0, phan_size*scale_factor, 1):
          for k in range(0, phan_size*scale_factor, 1):
            # Check if the voxel is True in the supersampled array
            if voxel_indices_super[i, j, k]:
              # If it is, set the corresponding downsampled voxel to True
              voxel_indices[i//scale_factor, j//scale_factor, k//scale_factor] = True

      # Remove all voxels outside cylinder radius phan_size/2 from centre of phantom
      mask = create_phantom_cylinder_mask(shape)
      voxel_indices = np.logical_and(voxel_indices, mask)

      phantom[chan_scat][voxel_indices] = scat_val
      phantom[chan_atten][voxel_indices] = atten_val

      metadata["inserts"].append({
        "material": insert_material,
        "shape": shape_type,
        "center": (center_x, center_y, center_z),
        "radius": radius,
        "is_anomaly": is_anomaly,
        "scat_val": scat_val,
        "atten_val": atten_val,
      })

    return phantom, metadata