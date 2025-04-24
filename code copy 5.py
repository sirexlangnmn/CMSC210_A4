import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load and preprocess the dataset
df = pd.read_csv("Dataset.csv")
df.columns = df.columns.str.strip()  # Remove accidental whitespace in headers

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
df['StressLevel'] = df['StressLevel'].map({
    'Low (1 - 2)': 0, 'Medium (3 - 4)': 1, 'High (5 - 6)': 2, 'Very High (7+)': 3
})
df['UseOfDevice'] = df['UseOfDevice'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})
df['SleepQuality'] = df['SleepQuality'].map({
    'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3
})
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature set and target
X = df[[
    'Gender', 'Age', 'OccupationType', 'CaffeineConsumption',
    'SmokingHabit', 'AlcoholConsumption', 'SleepingEnvironment',
    'AveSleep', 'ExerciseFrequency', 'StressLevel', 'UseOfDevice'
]]
y = df['SleepQuality']

# Class labels and names
class_names = ['Poor', 'Average', 'Good', 'Excellent']
all_labels = list(range(len(class_names)))

# Stratified 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Track best model
results = {}
best_model = None
best_accuracy = 0.0

# Run and evaluate models
for name, model in models.items():
    print("=" * 50)
    print(f"🔍 Evaluating Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0
    )
    print(f"{name} Classification Report:\n{report}")

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

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = name

# Summary
print("=" * 50)
print("📊 Model Performance Summary:")
for name, metrics in results.items():
    print(f"{name} Accuracy: {metrics['accuracy']:.4f}")

print(f"\n🏆 Best Model: {best_model} with Accuracy: {best_accuracy:.4f}")
