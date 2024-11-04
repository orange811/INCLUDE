import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load predictions
with open("predictions.json", "r") as f:
    predictions = json.load(f)

# Initialize counters
tp, fn, fp, tn = 0, 0, 0, 0

# Calculate TP, FN, FP, TN
for item in predictions:
    uid = item["uid"]
    predicted_label = item["predicted_label"]
    true_label = uid.split("_")[1].split(" ")[0]  # Extract the true label from uid

    if predicted_label == true_label:
        tp += 1
    else:
        # Check if the predicted label exists as a true label for other entries
        if any(
            pred["uid"].split("_")[1].split(" ")[0] == predicted_label
            for pred in predictions
        ):
            fp += 1  # Label exists in other categories, making it a False Positive
        else:
            tn += 1  # Label does not exist in this category or others, making it a True Negative
        fn += 1  # Also count as False Negative since it didn't match true_label

# Calculate metrics
accuracy = tp / (tp + fn + fp + tn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display results
print(f"True Positives (TP): {tp}")
print(f"False Negatives (FN): {fn}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
