import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\hey\OneDrive\Bureau\ARTD\Classic\Reults_MTTF\reliability_data.csv")
df.rename(columns={"MTTF": "MTTF (heures)", "MTTR": "MTTR (heures)"}, inplace=True)

if not all(col in df.columns for col in ["Composant", "MTTF (heures)", "MTTR (heures)"]):
    raise ValueError("Le fichier CSV doit contenir les colonnes : 'Composant', 'MTTF (heures)', 'MTTR (heures)'.")

#sim time
time_steps = 1000

# Init
state_probabilities = {}

for index, row in df.iterrows():
    mttf = float(row["MTTF (heures)"])
    mttr = float(row["MTTR (heures)"])
    
    if mttf == 0 or mttr == 0:
        continue  

    # Calcul des taux de transition
    lambda_failure = 1 / mttf
    mu_repair = 1 / mttr

    # Matrice de transition de Markov
    P = np.array([
        [1 - lambda_failure, lambda_failure, 0],  # S0 -> S1
        [0, 1 - mu_repair, mu_repair],  # S1 -> S2 ou retour à S0
        [0, 0, 1]  # S2 est un état absorbant (panne totale)
    ])

    # Init
    probs = np.zeros((time_steps, 3))
    probs[0] = [1, 0, 0]  
    for t in range(1, time_steps):
        probs[t] = np.dot(probs[t - 1], P)

    state_probabilities[row["Composant"]] = probs

first_component = df.iloc[0]["Composant"]
probs = state_probabilities[first_component]

plt.figure(figsize=(10, 5))
plt.plot(probs[:, 0], label="S0 (Opérationnel)", color="green")
plt.plot(probs[:, 1], label="S1 (Dégradé)", color="orange")
plt.plot(probs[:, 2], label="S2 (Panne)", color="red")
plt.xlabel("Temps (heures)")
plt.ylabel("Probabilité d'état")
plt.title(f"Évolution des États du Composant : {first_component} (Chaîne de Markov)")
plt.legend()
plt.grid()
plt.show()
