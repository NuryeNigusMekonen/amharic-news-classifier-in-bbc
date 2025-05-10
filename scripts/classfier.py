import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model and vectorizer
clf = joblib.load("amharic_news_classifier_300+.pkl")
vectorizer = joblib.load("tfidf_vectorizer_300+.pkl")

# Load and clean data
df = pd.read_csv("bbc_amharic_dataset_300+.csv")
df.dropna(subset=["text", "label"], inplace=True)
df["text"] = df["text"].apply(lambda x: re.sub(r"[፣።፤፥,!?\"']", "", str(x)).strip())

# Balance dataset
min_count = df['label'].value_counts().min()
df_balanced = pd.concat([
    df[df['label'] == label].sample(min_count, random_state=42)
    for label in df['label'].unique()
])

# Vectorize and split
X = vectorizer.transform(df_balanced["text"])
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
labels = sorted(df_balanced["label"].unique())

# ========== 📊 METRICS BAR CHART ==========
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(labels))
width = 0.25
data = {metric: [report[label][metric] for label in labels] for metric in metrics}

plt.figure(figsize=(10, 6))
plt.bar(x - width, data['precision'], width, label='Precision', color='skyblue')
plt.bar(x, data['recall'], width, label='Recall', color='orange')
plt.bar(x + width, data['f1-score'], width, label='F1-Score', color='green')
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("📊 Class-wise Metrics")
plt.grid(axis="y", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()

# ========== 📈 ACCURACY BAR CHART ==========
plt.figure(figsize=(4, 5))
plt.bar(["Accuracy"], [accuracy], color="purple")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("🎯 Overall Accuracy")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.show()

# ========== 🧩 CONFUSION MATRIX ==========
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧩 Confusion Matrix - Amharic News Classification")
plt.tight_layout()
plt.show()
