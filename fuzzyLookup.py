import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Step 1: Read the Data
file_path = 'NEW.xlsx'  # Replace with your file path
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit()

# Remove rows with missing values
data = data.dropna()

# Step 2: Define Membership Functions
heart_rate = ctrl.Antecedent(np.arange(40, 160, 1), 'heart_rate')
systolic_bp = ctrl.Antecedent(np.arange(70, 200, 1), 'systolic_bp')
resp_rate = ctrl.Antecedent(np.arange(5, 35, 1), 'resp_rate')
o2_sats = ctrl.Antecedent(np.arange(70, 105, 1), 'o2_sats')
temperature = ctrl.Antecedent(np.arange(30, 42, 0.1), 'temperature')

# Define membership functions
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
        # Validate inputs
        hr = np.clip(row['HEART_RATE'], 40, 159)
        sbp = np.clip(row['SYSTOLIC_BP'], 70, 199)
        rr = np.clip(row['RESP_RATE'], 5, 34)
        o2 = np.clip(row['O2_SATS'], 70, 104)
        temp = np.clip(row['TEMPERATURE'], 30, 41.9)
        
        event_simulation.input['heart_rate'] = hr
        event_simulation.input['systolic_bp'] = sbp
        event_simulation.input['resp_rate'] = rr
        event_simulation.input['o2_sats'] = o2
        event_simulation.input['temperature'] = temp
        
        event_simulation.compute()
        results.append(event_simulation.output['event'])
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        results.append(np.nan)

data['Predicted_Event'] = results

# Display the results
print(data)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['HEART_RATE'], data['SYSTOLIC_BP'], c=data['Predicted_Event'], cmap='viridis')
plt.colorbar(label='Event Likelihood')
plt.xlabel('Heart Rate')
plt.ylabel('Systolic BP')
plt.title('Event Likelihood based on Heart Rate and Systolic BP')
plt.show()