def process_threat_logs(log_data):
    # Example NLP feature extraction logic
    nlp_features = []
    for entry in log_data:
        nlp_features.append(entry["log"])  # Just extracting the log for simplicity
    return nlp_features

# Or if processing a single log:
def process_threat_logs(log_sample):
    # Example processing of a single log entry
    return [log_sample]  # Return a simple list with the log entry
