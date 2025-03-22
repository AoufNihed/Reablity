import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Génération de données factices
heures = np.arange(0, 24, 1)
tension = np.random.uniform(200, 230, 24)
courant = np.random.uniform(5, 20, 24)

plt.figure(figsize=(10, 5))
plt.plot(heures, tension, label="Tension (V)", marker='o', linestyle='-')
plt.plot(heures, courant, label="Courant (A)", marker='s', linestyle='--')

plt.xlabel("Heures")
plt.ylabel("Valeurs électriques")
plt.title("Évolution de la tension et du courant sur 24 heures")
plt.legend()
plt.grid(True)
plt.show()

# Génération de pannes aléatoires
pannes = np.random.randint(0, 5, 24)

plt.figure(figsize=(10, 5))
sns.barplot(x=heures, y=pannes, palette="Blues")

plt.xlabel("Heures")
plt.ylabel("Nombre de pannes")
plt.title("Histogramme des pannes détectées par heure")
plt.grid(axis='y')
plt.show()

charge = np.random.uniform(10, 90, 24)

plt.figure(figsize=(10, 5))
plt.plot(heures, charge, label="Charge (%)", marker='o', color='green')

plt.axhline(y=80, color='red', linestyle='--', label="Seuil de surcharge")
plt.axhline(y=20, color='orange', linestyle='--', label="Seuil de sous-utilisation")

plt.xlabel("Heures")
plt.ylabel("Charge (%)")
plt.title("Répartition de la charge électrique sur la journée")
plt.legend()
plt.grid(True)
plt.show()

harmoniques = np.random.uniform(0, 10, 24)

plt.figure(figsize=(10, 5))
plt.plot(heures, tension, label="Tension (V)", marker='o', linestyle='-')
plt.plot(heures, harmoniques, label="Harmoniques (%)", marker='s', linestyle='--', color='purple')

plt.xlabel("Heures")
plt.ylabel("Valeurs électriques")
plt.title("Évolution de la tension et des harmoniques")
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd

# Création d'une matrice de données aléatoires
data = np.random.uniform(0, 10, (24, 10))
df = pd.DataFrame(data, index=heures, columns=[f"Charge {i}" for i in range(1, 11)])

plt.figure(figsize=(10, 6))
sns.heatmap(df, cmap="coolwarm", linewidths=0.5)

plt.xlabel("Charge")
plt.ylabel("Heures")
plt.title("Carte thermique des niveaux de qualité")
plt.show()
