import pandas as pd
import json
from models.model_trainer import train_predictive_model
from nlp.parser import process_threat_logs
from kg.kg_builder import build_knowledge_graph
from visualizations import plot_feature_importance, plot_confusion_matrix, plot_predictions_vs_actual

# Load CAN bus data
can_data = pd.read_csv("data/can_logs.csv")
print(f"[✓] Loaded CAN data: {len(can_data)} records")

# Load threat intelligence
with open("data/threat_intel.json", "r") as f:
    threat_data = json.load(f)
print(f"[✓] Loaded threat intel: {len(threat_data)} entries")

# Step 1: Process threat logs with NLP
nlp_insights = process_threat_logs(threat_data)

# Step 2: Train/predict threats using CAN data
can_data, clf, y_test, y_pred = train_predictive_model(can_data, nlp_insights)

# Now call the visualization functions
plot_feature_importance(clf, can_data.drop(columns=['threat', 'predicted_threat']))
plot_confusion_matrix(y_test, y_pred)
plot_predictions_vs_actual(y_test, y_pred)

# Sample log to process and extract threat features
log_sample = "The attacker used CAN injection to spoof RPM data and trigger firmware exploits."
threat_features = process_threat_logs(log_sample)

# Step 3: Build/Update Knowledge Graph
build_knowledge_graph(can_data, nlp_insights)

print("[✓] Threat detection pipeline complete.")
