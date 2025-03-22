import numpy as np
import pandas as pd

def generate_data(n=50):
    """ Simule des données de tension, courant, charge, fréquence et harmoniques """
    np.random.seed(42)
    voltages = np.random.normal(220, 5, n)
    currents = np.random.uniform(10, 100, n)
    loads = np.random.uniform(50, 200, n)
    frequencies = np.random.normal(50, 0.2, n)
    harmonics = np.random.uniform(0, 10, n)

    return pd.DataFrame({
        "voltage": voltages,
        "current": currents,
        "load": loads,
        "frequency": frequencies,
        "harmonics": harmonics
    })
