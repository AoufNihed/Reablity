import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')

class ElectricQualityAgent:
    def __init__(self, voltage_thresholds=(210, 230), harmonic_limit=5):
        self.voltage_min, self.voltage_max = voltage_thresholds
        self.harmonic_limit = harmonic_limit
        self.data = []

    def monitor(self, voltage, harmonics):
        """ Analyse la tension et les harmoniques pour détecter des anomalies """
        status = "✅ Opération Normale"
        if voltage < self.voltage_min:
            status = "⚠️ Alerte: Sous-tension"
        elif voltage > self.voltage_max:
            status = "⚠️ Alerte: Surtension"
        if harmonics > self.harmonic_limit:
            status += " | 🔴 Harmoniques Élevées"

        self.data.append({"Voltage": voltage, "Harmonics": harmonics, "Status": status})
        print(f"Tension: {voltage}V, Harmoniques: {harmonics:.2f}% -> {status}")

    def generate_data(self, n=50):
        """ Simule des données de tension et d'harmoniques """
        np.random.seed(42)  # Fixe la seed pour des résultats reproductibles
        voltages = np.random.normal(220, 5, n)  # Moyenne = 220V, écart-type = 5
        harmonics = np.random.uniform(0, 10, n)  # Valeurs entre 0% et 10%
        return voltages, harmonics

    def visualize(self):
        """ Génère des graphiques pour analyser la qualité de l'électricité """
        df = pd.DataFrame(self.data)
        
        plt.figure(figsize=(10, 5))

        # 🔵 Graphique de la tension
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df["Voltage"], label="Tension (V)", color="blue")
        plt.axhline(self.voltage_max, linestyle="--", color="red", label="Seuil max")
        plt.axhline(self.voltage_min, linestyle="--", color="green", label="Seuil min")
        plt.legend()
        plt.title("Évolution de la Tension")

        # 🟣 Graphique des harmoniques
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df["Harmonics"], label="Harmoniques (%)", color="purple")
        plt.axhline(self.harmonic_limit, linestyle="--", color="orange", label="Seuil critique")
        plt.legend()
        plt.title("Évolution des Harmoniques")

        plt.tight_layout()
        plt.show()

    def export_data(self, filename="qualite_electricite.csv"):
        """ Exporte les données en CSV """
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"✅ Données exportées sous {filename}")

# 🔥 Exécution principale
if __name__ == "__main__":
    
    agent = ElectricQualityAgent()

    # 📡 Simulation des données
    voltages, harmonics = agent.generate_data(50)

    # 🔎 Surveillance et analyse
    for v, h in zip(voltages, harmonics):
        agent.monitor(v, h)

    # 📊 Visualisation des tendances
    agent.visualize()

    # 📤 Export des résultats
    agent.export_data()
