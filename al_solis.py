import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Load and preprocess the dataset
df = pd.read_csv("Dataset.csv")

# Map categorical values
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


# Full features and target
X = df[['Gender', 'Age', 'OccupationType', 'CaffeineConsumption', 'SmokingHabit', 'AlcoholConsumption',
          'SleepingEnvironment', 'AveSleep', 'ExerciseFrequency', 'StressLevel', 'UseOfDevice']]
y = df['SleepQuality']
class_names = ['Poor', 'Average', 'Good', 'Excellent']
all_labels = [0, 1, 2, 3]

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42, max_depth=4)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, rounded=True, feature_names=X.columns, class_names=class_names)
plt.title("Decision Tree")
plt.show()

# 2D Scatter Plot using kNN with SleepHours & Stress
features_2d = ['Age', 'StressLevel']
X_2d = df[features_2d]
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_2d_scaled, y)
y_pred_2d = knn_2d.predict(X_2d_scaled)

# Plot 2D scatter
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_pred_2d, cmap='viridis', edgecolor='k', s=80)
plt.xlabel('Age')
plt.ylabel('StressLevel')
plt.title('k-NN 2D Scatter Plot: Predicted Sleep Quality')
# Get unique predicted classes
unique_classes = np.unique(y_pred_2d)
legend_labels = [class_names[int(i)] for i in unique_classes]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

# 2D Scatter Plot for Naive Bayes
nb_2d = GaussianNB()
nb_2d.fit(X_2d_scaled, y)  # using Age and StressLevel scaled

y_pred_nb_2d = nb_2d.predict(X_2d_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_pred_nb_2d, cmap='spring', edgecolor='k', s=80)
plt.xlabel('Age (scaled)')
plt.ylabel('StressLevel (scaled)')
plt.title('Naive Bayes 2D Scatter Plot: Predicted Sleep Quality')

# Custom legend
unique_nb_classes = np.unique(y_pred_nb_2d)
legend_labels_nb = [class_names[int(i)] for i in unique_nb_classes]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels_nb, title="Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()


# Confusion Matrix - Decision Tree (Seaborn version)
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=all_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt, labels=all_labels, target_names=class_names, zero_division=0))

# k-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# Confusion Matrix - kNN (Seaborn version)
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=all_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.title("k-NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("k-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn, labels=all_labels, target_names=class_names, zero_division=0))

# Confusion Matrix - Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=all_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb, labels=all_labels, target_names=class_names, zero_division=0))

# Accuracy Comparison
print("Classifier Performance Comparison:")
print(f"Decision Tree Accuracy: {acc_dt:.2f}")
print(f"k-Nearest Neighbors Accuracy: {acc_knn:.2f}")
print(f"Naive Bayes Accuracy: {acc_nb:.2f}")

best_model = max(
    [("Decision Tree", acc_dt), ("k-NN", acc_knn), ("Naive Bayes", acc_nb)],
    key=lambda x: x[1]
)
print(f"Best Performer: {best_model[0]}")



