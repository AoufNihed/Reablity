import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

# 1. Simulation 
data = {
    'voltage': np.random.normal(230, 20, 1000),  
    'current': np.random.normal(30, 10, 1000),  
    'frequency': np.random.normal(50, 0.5, 1000), 
    'breaker_status': np.random.randint(0, 2, 1000),  # 0: f, 1: o
    'transformer_temp': np.random.normal(60, 5, 1000)  
}

df = pd.DataFrame(data)

# 2.seuils avec K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df[['voltage', 'current']])
df['fault_cluster'] = kmeans.labels_

# Déterminer les seuils de classification basés sur les centres des clusters
cluster_centers = kmeans.cluster_centers_
over_under_voltage_threshold = cluster_centers[:, 0].min()  # Plus bas centre
short_circuit_threshold = cluster_centers[:, 1].max()  # Plus haut centre

# Viz
plt.figure(figsize=(8, 6))
plt.scatter(df['voltage'], df['current'], c=df['fault_cluster'], cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centres des Clusters')
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Segmentation des défauts avec K-Means")
plt.legend()
plt.show()

def label_fault(row):
    if row['voltage'] < over_under_voltage_threshold or row['voltage'] > 250:
        return "Over/Under Voltage"
    elif row['current'] > short_circuit_threshold:
        return "Short Circuit"
    else:
        return "Normal"

df['fault_type'] = df.apply(label_fault, axis=1)

X = df[['voltage', 'current', 'frequency', 'breaker_status', 'transformer_temp']]
y = df['fault_type']

# Normalisation 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5.Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. eval
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# 7. Fonction prédiction en temps réel
def predict_fault(voltage, current, frequency, breaker_status, transformer_temp):
    input_data = np.array([[voltage, current, frequency, breaker_status, transformer_temp]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# 8. Export
df['predicted_fault'] = df.apply(lambda row: predict_fault(row['voltage'], row['current'], row['frequency'], row['breaker_status'], row['transformer_temp']), axis=1)
df.to_csv("fault_detection_results.csv", index=False)
print("Fichier CSV 'fault_detection_results.csv' créé avec les résultats des prédictions.")

# 9. Viz analytique
def plot_data():
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['voltage'], bins=30, color='blue', alpha=0.7)
    plt.title("Distribution de la Tension")
    
    plt.subplot(2, 2, 2)
    plt.hist(df['current'], bins=30, color='red', alpha=0.7)
    plt.title("Distribution du Courant")
    
    plt.subplot(2, 2, 3)
    plt.hist(df['frequency'], bins=30, color='green', alpha=0.7)
    plt.title("Distribution de la Fréquence")
    
    plt.subplot(2, 2, 4)
    plt.hist(df['transformer_temp'], bins=30, color='purple', alpha=0.7)
    plt.title("Température du Transformateur")
    
    plt.tight_layout()
    plt.show()

plot_data()
