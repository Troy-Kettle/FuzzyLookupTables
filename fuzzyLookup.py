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
heart_rate = ctrl.Antecedent(np.arange(0, 220, 1), 'heart_rate')
systolic_bp = ctrl.Antecedent(np.arange(0, 300, 1), 'systolic_bp')
resp_rate = ctrl.Antecedent(np.arange(0, 60, 1), 'resp_rate')
o2_sats = ctrl.Antecedent(np.arange(0, 100, 1), 'o2_sats')
temperature = ctrl.Antecedent(np.arange(20, 45, 0.1), 'temperature')

# Define membership functions with wider ranges
heart_rate['very_low'] = fuzz.trimf(heart_rate.universe, [0, 40, 60])
heart_rate['low'] = fuzz.trimf(heart_rate.universe, [40, 60, 80])
heart_rate['normal'] = fuzz.trimf(heart_rate.universe, [60, 80, 100])
heart_rate['high'] = fuzz.trimf(heart_rate.universe, [80, 100, 120])
heart_rate['very_high'] = fuzz.trimf(heart_rate.universe, [100, 160, 220])

systolic_bp['very_low'] = fuzz.trimf(systolic_bp.universe, [0, 70, 90])
systolic_bp['low'] = fuzz.trimf(systolic_bp.universe, [70, 90, 110])
systolic_bp['normal'] = fuzz.trimf(systolic_bp.universe, [90, 120, 140])
systolic_bp['high'] = fuzz.trimf(systolic_bp.universe, [130, 160, 180])
systolic_bp['very_high'] = fuzz.trimf(systolic_bp.universe, [160, 200, 300])

resp_rate['very_low'] = fuzz.trimf(resp_rate.universe, [0, 8, 12])
resp_rate['low'] = fuzz.trimf(resp_rate.universe, [8, 12, 16])
resp_rate['normal'] = fuzz.trimf(resp_rate.universe, [12, 16, 20])
resp_rate['high'] = fuzz.trimf(resp_rate.universe, [18, 25, 30])
resp_rate['very_high'] = fuzz.trimf(resp_rate.universe, [25, 40, 60])

o2_sats['very_low'] = fuzz.trimf(o2_sats.universe, [0, 80, 85])
o2_sats['low'] = fuzz.trimf(o2_sats.universe, [80, 85, 90])
o2_sats['normal'] = fuzz.trimf(o2_sats.universe, [85, 95, 100])
o2_sats['high'] = fuzz.trimf(o2_sats.universe, [95, 98, 100])

temperature['very_low'] = fuzz.trimf(temperature.universe, [20, 30, 34])
temperature['low'] = fuzz.trimf(temperature.universe, [30, 34, 36])
temperature['normal'] = fuzz.trimf(temperature.universe, [34, 36.5, 37.5])
temperature['high'] = fuzz.trimf(temperature.universe, [37, 38, 40])
temperature['very_high'] = fuzz.trimf(temperature.universe, [39, 41, 45])

# Define the output
event = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'event')
event['very_unlikely'] = fuzz.trimf(event.universe, [0, 0, 0.25])
event['unlikely'] = fuzz.trimf(event.universe, [0, 0.25, 0.5])
event['possible'] = fuzz.trimf(event.universe, [0.25, 0.5, 0.75])
event['likely'] = fuzz.trimf(event.universe, [0.5, 0.75, 1])
event['very_likely'] = fuzz.trimf(event.universe, [0.75, 1, 1])

# Step 3: Define Fuzzy Rules
rule1 = ctrl.Rule(heart_rate['very_high'] | systolic_bp['very_high'] | resp_rate['very_high'] | o2_sats['very_low'] | temperature['very_high'], event['very_likely'])
rule2 = ctrl.Rule(heart_rate['very_low'] | systolic_bp['very_low'] | resp_rate['very_low'] | temperature['very_low'], event['very_likely'])
rule3 = ctrl.Rule(heart_rate['normal'] & systolic_bp['normal'] & resp_rate['normal'] & o2_sats['normal'] & temperature['normal'], event['unlikely'])
rule4 = ctrl.Rule(heart_rate['high'] | systolic_bp['high'] | resp_rate['high'] | temperature['high'], event['likely'])
rule5 = ctrl.Rule(heart_rate['low'] | systolic_bp['low'] | resp_rate['low'] | o2_sats['low'] | temperature['low'], event['possible'])
rule6 = ctrl.Rule(o2_sats['high'], event['unlikely'])

# Create control system
event_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
event_simulation = ctrl.ControlSystemSimulation(event_ctrl)

# Step 4: Perform Fuzzy Inference
results = []

for index, row in data.iterrows():
    try:
        # Validate inputs
        hr = np.clip(row['HEART_RATE'], 0, 219)
        sbp = np.clip(row['SYSTOLIC_BP'], 0, 299)
        rr = np.clip(row['RESP_RATE'], 0, 59)
        o2 = np.clip(row['O2_SATS'], 0, 100)
        temp = np.clip(row['TEMPERATURE'], 20, 44.9)
        
        event_simulation.input['heart_rate'] = hr
        event_simulation.input['systolic_bp'] = sbp
        event_simulation.input['resp_rate'] = rr
        event_simulation.input['o2_sats'] = o2
        event_simulation.input['temperature'] = temp
        
        event_simulation.compute()
        results.append(event_simulation.output['event'])
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        # Assign a default value (e.g., 0.5 for 'possible') when computation fails
        results.append(0.5)

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