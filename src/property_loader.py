import json
import os
import random
import bisect

class PropertyLoader:
    def __init__(self, 
                 prop_data_path: str, 
                 atten_data_path: str,
                 prop_comp_data_path: str,
                 atten_comp_data_path: str):
        self.prop_data_path = prop_data_path
        self.atten_data_path = atten_data_path
        self.prop_comp_data_path = prop_comp_data_path # Compounds
        self.atten_comp_data_path = atten_comp_data_path # Compounds
        self.prop_data: dict | list | None = None
        self.atten_data: dict | list | None = None
        
        with open(self.prop_data_path, 'r') as f:
            self.prop_data = json.load(f)
        with open(self.atten_data_path, 'r') as f:
            self.atten_data = json.load(f)
        # Load compound data
        with open(self.prop_comp_data_path, 'r') as f:
            self.prop_data[0] |= json.load(f)[0]
            print("Property data loaded, length of prop_data: ", len(self.prop_data[0]))
        with open(self.atten_comp_data_path, 'r') as f:
            self.atten_data += json.load(f)
            print("Compound data loaded, length of atten_data: ", len(self.atten_data))

    def get_prop_data(self, name: str):
        raw_datum: dict = self.prop_data[0][name]
        return raw_datum
    
    def get_atten_data(self, name: str):
        raw_datum: list = next((item for item in self.atten_data if (name in item)), None)[name]
        datum: dict[float, float] = {} # Key: Energy (MeV), Value: Mass Attenuation Coefficient (cm^2/g)
        for row in raw_datum:
            datum[float(row[0])] = float(row[1])
        return datum
    
    def calc_electron_density(self, name: str, is_per_mole: bool = False):
        # Electron density given by the formula:
        # rho_e = (rho * N_A * Z) / A
        avogadro = 6.02214076e23 if not is_per_mole else 1.0
        prop_data = self.get_prop_data(name)

        def calc_rho_e(rho: float, Z: int, A: float):
            return (rho * avogadro * Z) / A
        
        total_rho_e = 0.0

        if not "composition" in prop_data:
            # Single element
            Z = int(name)
            rho = float(prop_data["density [g/cm^3]"])
            Z_A = float(prop_data["Z/A"])
            A = Z / Z_A
            total_rho_e = calc_rho_e(rho, Z, A)
        else:
            # Compound
            for _, (Z, proportion) in enumerate(prop_data["composition"].items()):
                rho = float(prop_data["density [g/cm^3]"])
                Z_A = float(self.get_prop_data(Z)["Z/A"]) # NIST technically gives us Z/A for the whole compound, but we'll compute it for each element
                A = int(Z) / Z_A
                total_rho_e += calc_rho_e(rho, int(Z), A) * float(proportion)

        return total_rho_e

    def calc_mass_atten_coeff_at_energy(self, name: str, energy: float):
        # Return NaN if energy is not in the range of the data
        if energy < 0.0 or energy > 20.0: # Hardcoded range of the data
            return float('NaN')

        atten_datum = self.get_atten_data(name)
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

    def get_all_material_names(self):
        # Get all atomic numbers from the property data
        return [atomic_number for atomic_number in self.prop_data[0].keys()]
    
    def get_random_material_name(self):
        return random.choice(self.get_all_material_names())