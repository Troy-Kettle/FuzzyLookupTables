import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Load the data
file_path = 'NEW.xlsx'  # Replace with your file path
try:
    data = pd.read_excel(file_path)
    print(f"Successfully loaded {len(data)} rows of data.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit()

# Remove rows with missing values
data_cleaned = data.dropna()
print(f"Removed {len(data) - len(data_cleaned)} rows with missing values.")

# Function to perform clustering and generate membership function boundaries
def generate_membership_boundaries(data, feature, n_clusters=5):
    # Reshape data for KMeans
    X = data[feature].values.reshape(-1, 1)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Get cluster centers and sort them
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    sorted_centers = np.sort(centers.flatten())
    
    # Generate membership function boundaries
    boundaries = {
        'very_low': [float(data[feature].min()), float(sorted_centers[0]), float(sorted_centers[1])],
        'low': [float(sorted_centers[0]), float(sorted_centers[1]), float(sorted_centers[2])],
        'normal': [float(sorted_centers[1]), float(sorted_centers[2]), float(sorted_centers[3])],
        'high': [float(sorted_centers[2]), float(sorted_centers[3]), float(sorted_centers[4])],
        'very_high': [float(sorted_centers[3]), float(sorted_centers[4]), float(data[feature].max())]
    }
    
    return boundaries

# Generate membership function boundaries
hr_boundaries = generate_membership_boundaries(data_cleaned, 'HEART_RATE')
sbp_boundaries = generate_membership_boundaries(data_cleaned, 'SYSTOLIC_BP')

# Visualize the generated membership functions
def plot_membership_functions(data, feature, boundaries):
    plt.figure(figsize=(10, 6))
    x = np.linspace(data[feature].min(), data[feature].max(), 1000)
    
    for name, points in boundaries.items():
        y = np.zeros_like(x)
        y[(x >= points[0]) & (x <= points[1])] = np.interp(x[(x >= points[0]) & (x <= points[1])], [points[0], points[1]], [0, 1])
        y[(x >= points[1]) & (x <= points[2])] = np.interp(x[(x >= points[1]) & (x <= points[2])], [points[1], points[2]], [1, 0])
        plt.plot(x, y, label=name)
    
    plt.title(f'{feature} Membership Functions')
    plt.xlabel(feature)
    plt.ylabel('Membership')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the generated membership functions
plot_membership_functions(data_cleaned, 'HEART_RATE', hr_boundaries)
plot_membership_functions(data_cleaned, 'SYSTOLIC_BP', sbp_boundaries)

# Save the generated boundaries to a JSON file
boundaries = {
    'HEART_RATE': hr_boundaries,
    'SYSTOLIC_BP': sbp_boundaries
}

with open('membership_boundaries.json', 'w') as f:
    json.dump(boundaries, f, indent=4)

print("Membership function boundaries have been saved to 'membership_boundaries.json'")