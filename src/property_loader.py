import json
import os
import random
import bisect

class PropertyLoader:
    def __init__(self, prop_data_path: str, atten_data_path: str):
        self.prop_data_path = prop_data_path
        self.atten_data_path = atten_data_path
        self.prop_data: dict | list | None = None
        self.atten_data: dict | list | None = None
        
        with open(self.prop_data_path, 'r') as f:
            self.prop_data = json.load(f)
        with open(self.atten_data_path, 'r') as f:
            self.atten_data = json.load(f)

    def get_prop_data_by_atomic_number(self, atomic_number: int):
        raw_datum: dict = self.prop_data[0][str(atomic_number)]
        datum: dict = {
            "Z": atomic_number, # bada bing bada boom let's just add it back in
            "symbol": raw_datum["symbol"],   
            "name": raw_datum["name"],
            "Z/A": float(raw_datum["Z/A"]),
            "density": float(raw_datum["density [g/cm^3]"]),
        }
        return datum
    
    def get_atten_data_by_atomic_number(self, atomic_number:int):
        raw_datum: list = next((item for item in self.atten_data if (str(atomic_number) in item)), None)[str(atomic_number)]
        datum: dict[float, float] = {} # Key: Energy (MeV), Value: Mass Attenuation Coefficient (cm^2/g)
        for row in raw_datum:
            datum[float(row[0])] = float(row[1])
        return datum
    
    def calc_electron_density_by_atomic_number(self, atomic_number: int, is_per_mole: bool = False):
        # Electron density given by the formula:
        # rho_e = (rho * N_A * Z) / A
        prop_data = self.get_prop_data_by_atomic_number(atomic_number)
        Z = prop_data["Z"]
        density = prop_data["density"]
        Z_A = prop_data["Z/A"]
        A = Z / Z_A
        avogadro = 6.02214076e23 if not is_per_mole else 1.0
        # Calculate electron density
        rho_e = (density * avogadro * Z) / A
        return rho_e

    def calc_mass_atten_coeff_by_atomic_number_at_energy(self, atomic_number: int, energy: float):
        # Return NaN if energy is not in the range of the data
        if energy < 0.0 or energy > 20.0: # Hardcoded range of the data
            return float('NaN')

        atten_datum = self.get_atten_data_by_atomic_number(atomic_number)
        energies = list(atten_datum.keys())
        index_left = bisect.bisect_left(energies, energy)
        index_right = index_left + 1
        if index_left == 0:
            index_right = 0
        elif index_right == len(energies):
            index_right = index_left - 1

        if index_left == index_right:
            return atten_datum[energies[index_left]]
        else:
            energy_left = energies[index_left]
            energy_right = energies[index_right]
            coeff_left = atten_datum[energy_left]
            coeff_right = atten_datum[energy_right]
            proportion = (energy - energy_left) / (energy_right - energy_left)
            coeff = coeff_left + proportion * (coeff_right - coeff_left)
            return coeff

    def get_all_atomic_numbers(self):
        # Get all atomic numbers from the property data
        return [int(atomic_number) for atomic_number in self.prop_data[0].keys()]
    
    def get_random_atomic_number(self):
        return random.choice(self.get_all_atomic_numbers())