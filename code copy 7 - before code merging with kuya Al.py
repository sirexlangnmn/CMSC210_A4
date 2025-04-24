"""
Final Model Evaluation and Meta-Analysis

Attributes Used for Classification:
The following 11 attributes were selected for predicting Sleep Quality. They were chosen based on their potential impact on sleep behavior:

1. Gender - Biological sex of the participant.
2. Age - Age in years.
3. OccupationType - Type of work, potentially affects routine and stress.
4. CaffeineConsumption - Whether the participant consumes caffeine (Yes/No).
5. SmokingHabit - Whether the participant smokes (Yes/No).
6. AlcoholConsumption - Whether the participant drinks alcohol (Yes/No).
7. SleepingEnvironment - The noise level of their sleeping area.
8. AveSleep - Average hours of sleep per night.
9. ExerciseFrequency - How often the participant exercises.
10. StressLevel - Self-reported stress level on a scale.
11. UseOfDevice - Frequency of using electronic devices before sleep.

Classification Algorithms Used:
- Decision Tree
- k-Nearest Neighbors
- Naive Bayes

Each model is evaluated using:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate  # For summary table

# Load and preprocess the dataset
df = pd.read_csv("Dataset.csv")
df.columns = df.columns.str.strip()

# Map categorical values to numerical
df['Gender'] = df['Gender'].map({'Prefer not to say': 0, 'Male': 1, 'Female': 2})
df['Age'] = df['Age'].astype(float)
df['OccupationType'] = df['OccupationType'].map({'Student': 0, 'Office Worker': 1, 'Shift Worker': 2, 'Self-Employed': 3, 'Retired': 4})
df['CaffeineConsumption'] = df['CaffeineConsumption'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
df['SmokingHabit'] = df['SmokingHabit'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
df['AlcoholConsumption'] = df['AlcoholConsumption'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
df['SleepingEnvironment'] = df['SleepingEnvironment'].map({'Quiet': 0, 'Moderate Noise': 1, 'Noisy': 2})
df['AveSleep'] = df['AveSleep'].astype(float)
df['ExerciseFrequency'] = df['ExerciseFrequency'].astype(float)
df['StressLevel'] = df['StressLevel'].map({'Low (1 - 2)': 0, 'Medium (3 - 4)': 1, 'High (5 - 6)': 2, 'Very High (7+)': 3})
df['UseOfDevice'] = df['UseOfDevice'].map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4})
df['SleepQuality'] = df['SleepQuality'].map({'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3})
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature set and target
X = df[['Gender', 'Age', 'OccupationType', 'CaffeineConsumption', 'SmokingHabit', 'AlcoholConsumption', 'SleepingEnvironment', 'AveSleep', 'ExerciseFrequency', 'StressLevel', 'UseOfDevice']]
y = df['SleepQuality']

# Class labels and names
class_names = ['Poor', 'Average', 'Good', 'Excellent']
all_labels = list(range(len(class_names)))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Models to evaluate
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}
best_model = None
best_accuracy = 0.0

for name, model in models.items():
    print("=" * 50)
    print(f"Evaluating Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, labels=all_labels, target_names=class_names, zero_division=0))

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = name

# Summary Table
print("=" * 50)
print("Model Performance Summary Table:")
table_data = []
for name, metrics in results.items():
    table_data.append([
        name,
        f"{metrics['accuracy']:.4f}",
        f"{metrics['precision']:.4f}",
        f"{metrics['recall']:.4f}",
        f"{metrics['f1_score']:.4f}"
    ])

print(tabulate(table_data, headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score"], tablefmt="grid"))

# Best Model
print(f"\n Best Model: {best_model} with Accuracy: {best_accuracy:.4f}")
