import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Data Generation
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Jour': np.arange(n_samples),
    'Demande': np.random.randint(2500, 6000, n_samples),
    'Production_Solaire': np.random.randint(500, 2000, n_samples),
    'Production_Eolienne': np.random.randint(300, 1500, n_samples),
    'Production_Hydro': np.random.randint(1000, 2500, n_samples),
    'Production_Gaz': np.random.randint(2000, 4000, n_samples),
    'Production_Vapeur': np.random.randint(500, 1500, n_samples),
    'Production_Cycle_Combine': np.random.randint(1000, 3000, n_samples)
})

data['Production_Totale'] = data[['Production_Solaire', 'Production_Eolienne', 'Production_Hydro',
                                  'Production_Gaz', 'Production_Vapeur', 'Production_Cycle_Combine']].sum(axis=1)

# Threshold Detection
mean_demande = data['Demande'].mean()
std_demande = data['Demande'].std()
seuil_bas = mean_demande - 2 * std_demande
seuil_haut = mean_demande + 2 * std_demande

data['Categorie_Demande'] = np.where(data['Demande'] < seuil_bas, 'Basse',
                                     np.where(data['Demande'] > seuil_haut, 'Élevée', 'Normale'))

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data[['Demande']])

# Anomaly Detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Anomalie'] = iso_forest.fit_predict(data[['Demande']])
data['Anomalie'] = data['Anomalie'].map({1: 'Normale', -1: 'Anomalie'})

# Demand Prediction Model
X = data[['Production_Solaire', 'Production_Eolienne', 'Production_Hydro', 'Production_Gaz',
          'Production_Vapeur', 'Production_Cycle_Combine']]
y = data['Demande']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

data['Demande_Predite'] = model.predict(X)

# AI Decision System
conditions = [
    (data['Demande'] < seuil_bas) | (data['Anomalie'] == 'Anomalie'),
    (data['Demande'] > seuil_haut),
    (data['Demande'] >= seuil_bas) & (data['Demande'] <= seuil_haut)
]
choix = ['Alerte: Baisse critique', 'Alerte: Surcharge', 'Opération Normale']
data['Decision_IA'] = np.select(conditions, choix, default='Opération Normale')

# Predict Next 7 Days
future_days = 7
future_productions = pd.DataFrame({
    'Jour': np.arange(n_samples, n_samples + future_days),
    'Production_Solaire': model.predict(X[-future_days:]) * 0.2,
    'Production_Eolienne': model.predict(X[-future_days:]) * 0.15,
    'Production_Hydro': model.predict(X[-future_days:]) * 0.25,
    'Production_Gaz': model.predict(X[-future_days:]) * 0.3,
    'Production_Vapeur': model.predict(X[-future_days:]) * 0.05,
    'Production_Cycle_Combine': model.predict(X[-future_days:]) * 0.05
})

# Save Results
data.to_csv("resultats_IA.csv", index=False)
future_productions.to_csv("future_productions.csv", index=False)

# Visualizations
# 1. Demand Forecasting
plt.figure(figsize=(12, 6))
plt.scatter(data['Jour'], data['Demande'], label='Demande Réelle', alpha=0.5)
plt.plot(data['Jour'], data['Demande_Predite'], color='red', label='Demande Prédite')
plt.axhline(seuil_bas, color='green', linestyle='dashed', label='Seuil Bas')
plt.axhline(seuil_haut, color='purple', linestyle='dashed', label='Seuil Haut')
plt.legend()
plt.xlabel('Jour')
plt.ylabel('Demande (MW)')
plt.title('Prévision de la Demande et Détection des Seuils')
plt.show()

# 2. Future Energy Production
future_productions.set_index('Jour').plot(kind='bar', stacked=True, figsize=(12, 6))
plt.xlabel('Jour')
plt.ylabel('Production (MW)')
plt.title('Prévisions de Production par Type Énergie')
plt.legend(title='Type de Production')
plt.show()

# 3. K-Means Clustering
plt.figure(figsize=(12, 6))
plt.scatter(data['Jour'], data['Demande'], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Jour')
plt.ylabel('Demande (MW)')
plt.title('Segmentation de la Demande par K-Means')
plt.colorbar(label='Cluster')
plt.show()

# 4. Anomaly Detection
plt.figure(figsize=(12, 6))
colors = {'Normale': 'blue', 'Anomalie': 'red'}
plt.scatter(data['Jour'], data['Demande'], c=data['Anomalie'].map(colors), alpha=0.6)
plt.xlabel('Jour')
plt.ylabel('Demande (MW)')
plt.title('Détection des Anomalies avec Isolation Forest')
plt.show()

# 5. AI Decision Distribution
plt.figure(figsize=(12, 6))
decision_counts = data['Decision_IA'].value_counts()
decision_counts.plot(kind='bar', color=['red', 'orange', 'green'])
plt.xlabel('Décision')
plt.ylabel('Nombre de jours')
plt.title('Distribution des Décisions de l’IA')
plt.show()
