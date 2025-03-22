import pandas as pd
import numpy as np

# Simulation de 200 événements de panne
np.random.seed(42)
n_samples = 200
components = ["Conducteur", "Isolateur", "Transformateur", "Disjoncteur", "Parafoudre"]
mttf_values = [40 * 365 * 24, 30 * 365 * 24, 25 * 365 * 24, 15 * 365 * 24, 20 * 365 * 24]  # en heures
mttr_values = [12, 6, 48, 4, 3]  # en heures

data = []
for _ in range(n_samples):
    component = np.random.choice(components)
    idx = components.index(component)
    
    # Génération du temps avant défaillance
    time_to_failure = -mttf_values[idx] * np.log(np.random.rand())
    
    # Ajout des valeurs
    data.append([component, mttf_values[idx], mttr_values[idx], time_to_failure])

# Création du DataFrame
df = pd.DataFrame(data, columns=["Composant", "MTTF", "MTTR", "Temps_Avant_Defaillance"])
df.to_csv("reliability_data.csv", index=False)

print("Fichier CSV : reliability_data.csv")
