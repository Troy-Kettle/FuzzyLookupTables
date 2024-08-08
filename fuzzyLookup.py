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
    """Define membership functions for Heart Rate and Systolic Blood Pressure."""
    hr_boundaries = boundaries['HEART_RATE']
    sbp_boundaries = boundaries['SYSTOLIC_BP']

    heart_rate = ctrl.Antecedent(np.arange(0, 220, 1), 'heart_rate')
    systolic_bp = ctrl.Antecedent(np.arange(0, 300, 1), 'systolic_bp')
    event = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'event')

    for category in ['very_low', 'low', 'normal', 'high', 'very_high']:
        heart_rate[category] = fuzz.trimf(heart_rate.universe, hr_boundaries[category])
        systolic_bp[category] = fuzz.trimf(systolic_bp.universe, sbp_boundaries[category])

    event['very_low'] = fuzz.trimf(event.universe, [0, 0, 0.25])
    event['low'] = fuzz.trimf(event.universe, [0, 0.25, 0.5])
    event['medium'] = fuzz.trimf(event.universe, [0.25, 0.5, 0.75])
    event['high'] = fuzz.trimf(event.universe, [0.5, 0.75, 1])
    event['very_high'] = fuzz.trimf(event.universe, [0.75, 1, 1])

    return heart_rate, systolic_bp, event

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

def define_fuzzy_rules(heart_rate, systolic_bp, event):
    """Define fuzzy rules for the system."""
    rule1 = ctrl.Rule(heart_rate['very_high'] | systolic_bp['very_high'], event['very_high'])
    rule2 = ctrl.Rule(heart_rate['very_low'] | systolic_bp['very_low'], event['high'])
    rule3 = ctrl.Rule(heart_rate['normal'] & systolic_bp['normal'], event['very_low'])
    rule4 = ctrl.Rule(heart_rate['high'] | systolic_bp['high'], event['medium'])
    rule5 = ctrl.Rule(heart_rate['low'] | systolic_bp['low'], event['low'])

    return ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

def validate_input(row):
    """Validate input data."""
    valid_ranges = {
        'HEART_RATE': (0, 220),
        'SYSTOLIC_BP': (0, 300)
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
        
        event_simulation.compute()
        return event_simulation.output['event']
    except Exception as e:
        logging.warning(f"Error processing row: {e}")
        return 0.5  # Assign medium risk for errors


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
    corr_columns = ['HEART_RATE', 'SYSTOLIC_BP', 'Predicted_Event']
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
        heart_rate, systolic_bp, event = define_membership_functions(boundaries)
        
        # Visualize membership functions
        visualize_membership_functions(heart_rate, systolic_bp)
        
        # Define fuzzy rules
        event_ctrl = define_fuzzy_rules(heart_rate, systolic_bp, event)
        
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
