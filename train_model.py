import pandas as pd
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv("job_role_prediction_dataset.csv")

# Encode the target variable 'JobRole' into numerical values
label_encoder = LabelEncoder()
df['JobRoleEncoded'] = label_encoder.fit_transform(df['JobRole'])

# Combine all keyword columns into a single 'keywords' column to apply vectorization
df['keywords'] = df[['Keyword1', 'Keyword2', 'Keyword3', 'Keyword4', 'Keyword5']].agg(' '.join, axis=1)

# Vectorization using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['keywords'])

# Target variable
y = df['JobRoleEncoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance Plot
importances = rf_classifier.feature_importances_
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[indices], align='center')
plt.yticks(range(len(importances)), [f'Feature {i+1}' for i in indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance - Random Forest')
plt.savefig('feature_importance.png')
plt.close()

# ROC Curve
# Note: multiclass ROC is complex, this code assumes binary or handles it in a specific way that might error if multiclass without specific handling (ovr).
# However, I will write the code exactly as provided by the user for now.
# One potential issue in the user code: `roc_curve` with `pos_label=1` suggests binary classification or it will fail on multiclass.
# The previous visualization code suggests multiple classes (heatmap labels).
# I'll stick to the user's code but keep an eye out for errors.

fpr, tpr, thresholds = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Save the model and artifacts
joblib.dump(rf_classifier, 'rf_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and artifacts saved to disk.")
