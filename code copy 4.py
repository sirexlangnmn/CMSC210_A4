import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("Dataset2.csv")

# Clean column names and trim whitespaces
df.columns = df.columns.str.strip()

# Set target
target_column = "SleepQuality"
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# Encode target column
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
class_names = y_encoder.classes_

# Stratified split to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Store results
results = {}
best_model_name = ""
best_model_accuracy = 0.0

# Evaluate each model
for name, model in models.items():
    print("=" * 50)
    print(f"🔍 Evaluating Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(f"{name} Classification Report:\n{report}")

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    if acc > best_model_accuracy:
        best_model_accuracy = acc
        best_model_name = name

# Summary
print("=" * 50)
print("📊 Classifier Performance Comparison:")
for name, metrics in results.items():
    print(f"{name} Accuracy: {metrics['accuracy']:.4f}")

print(f"\n🏆 Best Performer: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
