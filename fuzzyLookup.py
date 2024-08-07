import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load data from Excel file and clean it."""
    try:
        data = pd.read_excel(file_path)
        logging.info(f"Successfully loaded {len(data)} rows of data.")
        data_cleaned = data.dropna()
        logging.info(f"Removed {len(data) - len(data_cleaned)} rows with missing values.")
        return data_cleaned
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found. Please check the file path.")
        raise

def load_membership_boundaries(file_path):
    """Load membership boundaries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found. Please check the file path.")
        raise

def define_membership_functions(boundaries):
    """Define membership functions for all variables."""
    hr_boundaries = boundaries['HEART_RATE']
    sbp_boundaries = boundaries['SYSTOLIC_BP']

    heart_rate = ctrl.Antecedent(np.arange(0, 220, 1), 'heart_rate')
    systolic_bp = ctrl.Antecedent(np.arange(0, 300, 1), 'systolic_bp')
    resp_rate = ctrl.Antecedent(np.arange(0, 60, 1), 'resp_rate')
    o2_sats = ctrl.Antecedent(np.arange(0, 100, 1), 'o2_sats')
    temperature = ctrl.Antecedent(np.arange(20, 45, 0.1), 'temperature')

    for category in ['very_low', 'low', 'normal', 'high', 'very_high']:
        heart_rate[category] = fuzz.trimf(heart_rate.universe, hr_boundaries[category])
        systolic_bp[category] = fuzz.trimf(systolic_bp.universe, sbp_boundaries[category])

    # Define membership functions for other variables (unchanged)
    resp_rate['very_low'] = fuzz.trapmf(resp_rate.universe, [0, 0, 8, 10])
    resp_rate['low'] = fuzz.trimf(resp_rate.universe, [8, 12, 16])
    resp_rate['normal'] = fuzz.trimf(resp_rate.universe, [12, 16, 20])
    resp_rate['high'] = fuzz.trimf(resp_rate.universe, [18, 25, 32])
    resp_rate['very_high'] = fuzz.trapmf(resp_rate.universe, [30, 40, 60, 60])

    o2_sats['very_low'] = fuzz.trapmf(o2_sats.universe, [0, 0, 80, 85])
    o2_sats['low'] = fuzz.trimf(o2_sats.universe, [80, 85, 90])
    o2_sats['normal'] = fuzz.trimf(o2_sats.universe, [88, 95, 100])
    o2_sats['high'] = fuzz.trapmf(o2_sats.universe, [98, 100, 100, 100])

    temperature['very_low'] = fuzz.trapmf(temperature.universe, [20, 20, 32, 34])
    temperature['low'] = fuzz.trimf(temperature.universe, [32, 35, 36])
    temperature['normal'] = fuzz.trimf(temperature.universe, [35.5, 36.5, 37.5])
    temperature['high'] = fuzz.trimf(temperature.universe, [37, 38, 39])
    temperature['very_high'] = fuzz.trapmf(temperature.universe, [38.5, 40, 45, 45])

    event = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'event')
    event['very_low'] = fuzz.trimf(event.universe, [0, 0, 0.25])
    event['low'] = fuzz.trimf(event.universe, [0, 0.25, 0.5])
    event['medium'] = fuzz.trimf(event.universe, [0.25, 0.5, 0.75])
    event['high'] = fuzz.trimf(event.universe, [0.5, 0.75, 1])
    event['very_high'] = fuzz.trimf(event.universe, [0.75, 1, 1])

    return heart_rate, systolic_bp, resp_rate, o2_sats, temperature, event

def visualize_membership_functions(heart_rate, systolic_bp):
    """Visualize membership functions for Heart Rate and Blood Pressure."""
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))

    heart_rate.view(ax=ax0)
    ax0.set_title('Heart Rate Membership Functions')
    ax0.legend()

    systolic_bp.view(ax=ax1)
    ax1.set_title('Systolic Blood Pressure Membership Functions')
    ax1.legend()

    plt.tight_layout()
    plt.show()

def define_fuzzy_rules(heart_rate, systolic_bp, resp_rate, o2_sats, temperature, event):
    """Define fuzzy rules for the system."""
    rule1 = ctrl.Rule(heart_rate['very_high'] | systolic_bp['very_high'] | resp_rate['very_high'] | o2_sats['very_low'] | temperature['very_high'], event['very_high'])
    rule2 = ctrl.Rule(heart_rate['very_low'] | systolic_bp['very_low'] | resp_rate['very_low'] | temperature['very_low'], event['high'])
    rule3 = ctrl.Rule(heart_rate['normal'] & systolic_bp['normal'] & resp_rate['normal'] & o2_sats['normal'] & temperature['normal'], event['very_low'])
    rule4 = ctrl.Rule(heart_rate['high'] | systolic_bp['high'] | resp_rate['high'] | temperature['high'], event['medium'])
    rule5 = ctrl.Rule(heart_rate['low'] | systolic_bp['low'] | resp_rate['low'] | o2_sats['low'] | temperature['low'], event['low'])
    rule6 = ctrl.Rule(o2_sats['high'], event['very_low'])

    return ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])

def validate_input(row):
    """Validate input data."""
    valid_ranges = {
        'HEART_RATE': (0, 220),
        'SYSTOLIC_BP': (0, 300),
        'RESP_RATE': (0, 60),
        'O2_SATS': (0, 100),
        'TEMPERATURE': (20, 45)
    }
    
    for column, (min_val, max_val) in valid_ranges.items():
        if not min_val <= row[column] <= max_val:
            raise ValueError(f"Invalid {column}: {row[column]}")

def process_row(row, event_simulation):
    """Process a single row of data."""
    try:
        validate_input(row)
        
        event_simulation.input['heart_rate'] = row['HEART_RATE']
        event_simulation.input['systolic_bp'] = row['SYSTOLIC_BP']
        event_simulation.input['resp_rate'] = row['RESP_RATE']
        event_simulation.input['o2_sats'] = row['O2_SATS']
        event_simulation.input['temperature'] = row['TEMPERATURE']
        
        event_simulation.compute()
        return event_simulation.output['event']
    except Exception as e:
        logging.warning(f"Error processing row: {e}")
        return 0.5  # Assign medium risk for errors

def perform_fuzzy_inference(data, event_ctrl):
    """Perform fuzzy inference on the data."""
    event_simulation = ctrl.ControlSystemSimulation(event_ctrl)
    tqdm.pandas(desc="Processing data")
    data['Predicted_Event'] = data.progress_apply(lambda row: process_row(row, event_simulation), axis=1)
    return data

def categorize_risk(value):
    """
    Categorize risk based on the predicted event value.
    
    Args:
        value (float): Predicted event value between 0 and 1.
    
    Returns:
        str: Risk category ('Very Low', 'Low', 'Medium', 'High', or 'Very High').
    """
    if value < 0.2:
        return 'Very Low'
    elif value < 0.4:
        return 'Low'
    elif value < 0.6:
        return 'Medium'
    elif value < 0.8:
        return 'High'
    else:
        return 'Very High'

def analyze_results(data):
    """Analyze and visualize the results."""
    print("\nAnalysis of Results:")
    print(data['Predicted_Event'].describe())

    data['Risk_Category'] = data['Predicted_Event'].apply(categorize_risk)

    print("\nRisk Distribution:")
    risk_distribution = data['Risk_Category'].value_counts(normalize=True) * 100
    print(risk_distribution)

    # Visualizations
    plt.figure(figsize=(16, 12))

    # Scatter plot
    plt.subplot(221)
    scatter = plt.scatter(data['HEART_RATE'], data['SYSTOLIC_BP'], 
                          c=data['Predicted_Event'], cmap='RdYlGn_r', alpha=0.5)
    plt.colorbar(scatter, label='Event Risk')
    plt.xlabel('Heart Rate')
    plt.ylabel('Systolic BP')
    plt.title('Event Risk based on Heart Rate and Systolic BP')

    # Risk distribution
    plt.subplot(222)
    sns.barplot(x=risk_distribution.index, y=risk_distribution.values)
    plt.xlabel('Risk Category')
    plt.ylabel('Percentage')
    plt.title('Distribution of Risk Categories')
    plt.xticks(rotation=45)

    # Correlation heatmap
    plt.subplot(223)
    corr_columns = ['HEART_RATE', 'SYSTOLIC_BP', 'RESP_RATE', 'O2_SATS', 'TEMPERATURE', 'Predicted_Event']
    correlations = data[corr_columns].corr()
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')

    # Box plot
    plt.subplot(224)
    sns.boxplot(x='Risk_Category', y='Predicted_Event', data=data, order=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    plt.title('Distribution of Predicted Event by Risk Category')

    plt.tight_layout()
    plt.show()

    print("\nCorrelation with Predicted Event Risk:")
    print(correlations['Predicted_Event'].sort_values(ascending=False))

def main():
    try:
        # Load data
        data = load_data('NEW.xlsx')
        
        # Load membership boundaries
        boundaries = load_membership_boundaries('membership_boundaries.json')
        
        # Define membership functions
        heart_rate, systolic_bp, resp_rate, o2_sats, temperature, event = define_membership_functions(boundaries)
        
        # Visualize membership functions
        visualize_membership_functions(heart_rate, systolic_bp)
        
        # Define fuzzy rules
        event_ctrl = define_fuzzy_rules(heart_rate, systolic_bp, resp_rate, o2_sats, temperature, event)
        
        # Perform fuzzy inference
        results = perform_fuzzy_inference(data, event_ctrl)
        
        # Analyze results
        analyze_results(results)
        
        # Save results
        results.to_csv('risk_assessment_results.csv', index=False)
        logging.info("Results saved to 'risk_assessment_results.csv'")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()