import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the Data
file_path = 'NEW.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Remove rows with missing values
data = data.dropna()

# Step 2: Define Membership Functions
heart_rate = ctrl.Antecedent(np.arange(40, 160, 1), 'heart_rate')
systolic_bp = ctrl.Antecedent(np.arange(70, 200, 1), 'systolic_bp')
resp_rate = ctrl.Antecedent(np.arange(5, 35, 1), 'resp_rate')
o2_sats = ctrl.Antecedent(np.arange(70, 105, 1), 'o2_sats')
temperature = ctrl.Antecedent(np.arange(30, 42, 0.1), 'temperature')

# Define membership functions with adjusted ranges
heart_rate['low'] = fuzz.trimf(heart_rate.universe, [40, 60, 80])
heart_rate['normal'] = fuzz.trimf(heart_rate.universe, [80, 100, 120])
heart_rate['high'] = fuzz.trimf(heart_rate.universe, [120, 140, 160])

systolic_bp['low'] = fuzz.trimf(systolic_bp.universe, [70, 85, 100])
systolic_bp['normal'] = fuzz.trimf(systolic_bp.universe, [100, 120, 140])
systolic_bp['high'] = fuzz.trimf(systolic_bp.universe, [140, 160, 200])

resp_rate['low'] = fuzz.trimf(resp_rate.universe, [5, 10, 15])
resp_rate['normal'] = fuzz.trimf(resp_rate.universe, [15, 20, 25])
resp_rate['high'] = fuzz.trimf(resp_rate.universe, [25, 30, 35])

o2_sats['low'] = fuzz.trimf(o2_sats.universe, [70, 80, 90])
o2_sats['normal'] = fuzz.trimf(o2_sats.universe, [90, 95, 100])
o2_sats['high'] = fuzz.trimf(o2_sats.universe, [100, 102, 105])

temperature['low'] = fuzz.trimf(temperature.universe, [30, 32, 34])
temperature['normal'] = fuzz.trimf(temperature.universe, [34, 36.5, 38])
temperature['high'] = fuzz.trimf(temperature.universe, [38, 40, 42])

# Define the output
event = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'event')
event['unlikely'] = fuzz.trimf(event.universe, [0, 0, 0.5])
event['possible'] = fuzz.trimf(event.universe, [0.25, 0.5, 0.75])
event['likely'] = fuzz.trimf(event.universe, [0.5, 1, 1])

# Visualize the membership functions
heart_rate.view()
systolic_bp.view()
resp_rate.view()
o2_sats.view()
temperature.view()
event.view()

# Step 3: Define Fuzzy Rules
rule1 = ctrl.Rule(heart_rate['high'] & systolic_bp['high'] & resp_rate['high'] & o2_sats['low'] & temperature['high'], event['likely'])
rule2 = ctrl.Rule(heart_rate['low'] & systolic_bp['low'] & resp_rate['low'] & o2_sats['high'] & temperature['low'], event['unlikely'])
rule3 = ctrl.Rule(heart_rate['normal'] & systolic_bp['normal'] & resp_rate['normal'] & o2_sats['normal'] & temperature['normal'], event['possible'])
rule4 = ctrl.Rule(heart_rate['normal'] & systolic_bp['high'] & resp_rate['normal'] & o2_sats['normal'] & temperature['normal'], event['possible'])
rule5 = ctrl.Rule(heart_rate['low'] & systolic_bp['normal'] & resp_rate['low'] & o2_sats['high'] & temperature['normal'], event['unlikely'])
rule6 = ctrl.Rule(heart_rate['high'] & systolic_bp['normal'] & resp_rate['high'] & o2_sats['low'] & temperature['high'], event['likely'])

# Create control system
event_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
event_simulation = ctrl.ControlSystemSimulation(event_ctrl)

# Step 4: Perform Fuzzy Inference
results = []

for index, row in data.iterrows():
    try:
        # Validate inputs to ensure they are within expected ranges
        hr = max(min(row['HEART_RATE'], 159), 39)
        sbp = max(min(row['SYSTOLIC_BP'], 199), 69)
        rr = max(min(row['RESP_RATE'], 34), 4)
        o2 = max(min(row['O2_SATS'], 104), 69)
        temp = max(min(row['TEMPERATURE'], 41.9), 29.9)
        
        event_simulation.input['heart_rate'] = hr
        event_simulation.input['systolic_bp'] = sbp
        event_simulation.input['resp_rate'] = rr
        event_simulation.input['o2_sats'] = o2
        event_simulation.input['temperature'] = temp
        
        event_simulation.compute()
        results.append(event_simulation.output['event'])
    except Exception as e:
        print(f"Error with inputs {hr}, {sbp}, {rr}, {o2}, {temp}: {e}")
        results.append(np.nan)  # Handle error case by appending NaN

data['Predicted_Event'] = results

# Display the results
print(data)

# Optional: Visualize the Results
X, Y = np.meshgrid(np.arange(40, 160, 1), np.arange(70, 200, 1))
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        try:
            event_simulation.input['heart_rate'] = X[i, j]
            event_simulation.input['systolic_bp'] = Y[i, j]
            event_simulation.input['resp_rate'] = 19  # Use a default or average value
            event_simulation.input['o2_sats'] = 95    # Use a default or average value
            event_simulation.input['temperature'] = 36.5  # Use a default or average value
            
            event_simulation.compute()
            Z[i, j] = event_simulation.output['event']
        except Exception as e:
            print(f"Error with inputs {X[i, j]}, {Y[i, j]}, 19, 95, 36.5: {e}")
            Z[i, j] = np.nan  # Handle error case by assigning NaN

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Heart Rate')
ax.set_ylabel('Systolic BP')
ax.set_zlabel('Event Likelihood')

plt.show()
