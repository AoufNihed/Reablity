import pandas as pd
import numpy as np
from scipy.stats import expon

# Load the dataset
file_path = r"C:\Users\hey\OneDrive\Bureau\ARTD\Classic\Reults_MTTF\reliability_data.csv"
df = pd.read_csv(file_path)

def compute_failure_probability(mttf, time):
    """Calculate failure probability using an exponential distribution."""
    return 1 - np.exp(-time / mttf)

# Compute failure probabilities for each component
df["Failure_Probability"] = df.apply(lambda row: compute_failure_probability(row["MTTF"], row["Temps_Avant_Defaillance"]), axis=1)

# Quantitative FTA: Compute System Failure Probability
system_failure_probability = 1 - np.prod(1 - df["Failure_Probability"].values)
print(f"System Failure Probability: {system_failure_probability:.4f}")

# Boolean Logic FTA: Example Using OR Logic
boolean_result = np.max(df["Failure_Probability"].values)  # Assuming OR gate logic
print(f"Boolean Logic FTA Result: {boolean_result:.4f}")

# Fuzzy FTA (Example with Fuzzy Probabilities)
def fuzzy_failure_probability(mttf, alpha=0.1):
    """Estimate failure probability using a fuzzy approach."""
    lower_bound = compute_failure_probability(mttf, (1 - alpha) * df["Temps_Avant_Defaillance"].mean())
    upper_bound = compute_failure_probability(mttf, (1 + alpha) * df["Temps_Avant_Defaillance"].mean())
    return (lower_bound, upper_bound)

df["Fuzzy_Failure_Probability"] = df["MTTF"].apply(lambda x: fuzzy_failure_probability(x))
print(df[["Composant", "Fuzzy_Failure_Probability"]])
# Export fuzzy failure probability data
df[["Composant", "Fuzzy_Failure_Probability"]].to_csv("fuzzy_failure_data.csv", index=False)

print("Fuzzy failure probability data exported successfully.")

