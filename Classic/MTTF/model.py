import numpy as np

# Définition des paramètres de la ligne (exemple)
R = 0.2  
L = 0.5e-3  
C = 200e-9 
G = 0  

f = 50  
omega = 2 * np.pi * f  

# Calcul des paramètres du modèle π
Z = R + 1j * omega * L
Y = G + 1j * omega * C

# Longueur de la ligne en km
Ligne_L = 100

# Constante de propagation
gamma = np.sqrt(Z * Y)
Zc = np.sqrt(Z / Y)

# Calcul des tensions et courants avec le modèle π
def ligne_pi(V1, I1, L=Ligne_L):
    V2 = V1 * np.cosh(gamma * L) - I1 * Zc * np.sinh(gamma * L)
    I2 = I1 * np.cosh(gamma * L) - (V1 / Zc) * np.sinh(gamma * L)
    return V2, I2

# Test avec une tension d'entrée de 230V et un courant de 10A
V1_test, I1_test = 230, 10
V2_test, I2_test = ligne_pi(V1_test, I1_test)

print(f"Tension en sortie: {V2_test:.2f} V")
print(f"Courant en sortie: {I2_test:.2f} A")
