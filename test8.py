import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Load the trained model
rf_model_filename = "./new-rf-8class-model.pkl"
with open(rf_model_filename, "rb") as model_file:
    model = pickle.load(model_file)

# Define the input CSV file for testing
test_file = './Test/processed_output.csv'  # Replace with the path to your test CSV file

# Read the test data
test_data = pd.read_csv(test_file)

# Define the columns for input features (X)
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

# Scale the test data
scaler = StandardScaler()
test_data[X_columns] = scaler.fit_transform(test_data[X_columns])

# Define the binary classification mapping
dict_7classes = {
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'BenignTraffic': 'Benign',
    'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web',
    'XSS': 'Web',
    'Uploading_Attack': 'Web',
    'SqlInjection': 'Web',
    'CommandInjection': 'Web',
    'DictionaryBruteForce': 'BruteForce'
}

# Make predictions
prediction = model.predict(test_data[X_columns])
predictions = [str(x) for x in prediction]

# Map true labels
true_labels = [dict_7classes[str(x)] for x in test_data['label']]

# Prepare DataFrame for true and predicted labels
output_df = pd.DataFrame({
    'True_Label': true_labels,
    'Predicted_Label': predictions
})

# Output the predictions to CSV
output_file = 'predictions.csv'  # Replace with the desired output file path
output_df.to_csv(output_file, index=False)

# Calculate and display relevant metrics
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average='macro', zero_division=1)
precision = precision_score(true_labels, predictions, average='macro', zero_division=1)
f1 = f1_score(true_labels, predictions, average='macro')

print("Predictions saved to", output_file)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
