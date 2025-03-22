import numpy as np
import matplotlib.pyplot as plt

# Définition des paramètres
lambda_failure = 0.0001  # Taux de défaillance 
mu_repair = 0.01         # Taux de réparation 
delta_t = 1              # Pas de temps
time_steps = 1000        # Nombre d'itérations de la simulation

# Matrice de transition
P = np.array([
    [1 - lambda_failure * delta_t, lambda_failure * delta_t, 0],  # État S0 (Opérationnel)
    [0, 1 - mu_repair * delta_t, mu_repair * delta_t],  # État S1 (Dégradé)
    [0, 0, 1]  # État S2 (En panne, irréversible)
])

# Init
state_probabilities = np.zeros((time_steps, 3))
state_probabilities[0] = [1, 0, 0] 

for t in range(1, time_steps):
    state_probabilities[t] = np.dot(state_probabilities[t - 1], P)

plt.figure(figsize=(10, 5))
plt.plot(state_probabilities[:, 0], label="S0 (Opérationnel)", color="green")
plt.plot(state_probabilities[:, 1], label="S1 (Dégradé)", color="orange")
plt.plot(state_probabilities[:, 2], label="S2 (En Panne)", color="red")
plt.xlabel("Temps (cycles)")
plt.ylabel("Probabilité d'état")
plt.title("Évolution des États du Système (Chaînes de Markov)")
plt.legend()
plt.grid()
plt.show()
