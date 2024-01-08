import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Load the trained model
rf_model_filename = "./Models-34class/new-rf-34class-model.pkl"
with open(rf_model_filename, "rb") as model_file:
    model = pickle.load(model_file)

# Define the input CSV file for testing
test_file = './filtered_data.csv'  # Replace with the path to your test CSV file

# Read the test data
test_data = pd.read_csv(test_file)

# Define the columns for input features (X)
X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight',
]

scaler=StandardScaler()
test_data[X_columns] = scaler.fit_transform(test_data[X_columns])


# Make predictions
prediction = model.predict(test_data[X_columns])
predictions = prediction


labels = test_data['label']
true_labels = labels



# Calculate and display relevant metrics
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average='macro', zero_division=1)
precision = precision_score(true_labels, predictions, average='macro', zero_division=1)
f1 = f1_score(true_labels, predictions, average='macro')

# Output the predictions and metrics
output_file = 'predictions.csv'  # Replace with the desired output file path
test_data['Predicted_Label'] = predictions
test_data.to_csv(output_file, index=False)

print("Predictions saved to", output_file)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
