import os
import re
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#updated from git now in vscode
# === Folder paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# === File paths ===
CSV_PATH = os.path.join(DATA_DIR, "bbc_amharic_dataset_300+.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "amharic_news_classifier_300+.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer_300+.pkl")

# === Load and clean dataset ===
print(" Loading dataset...")
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["text", "label"], inplace=True)
df["text"] = df["text"].apply(lambda x: re.sub(r"[፣።፤፥,!?\"']", "", str(x)).strip())

# === Balance dataset === as 120 headlines and 90 headlines available for different catagories 
min_class = df["label"].value_counts().min()
df_balanced = pd.concat([
    df[df["label"] == label].sample(min_class, random_state=42)
    for label in df["label"].unique()
])
print(f" Balanced dataset. Samples per class: {min_class}")

# === TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True) #this transforms text to numerical value that logistic regression can understand.
#max_features=8000
#Limits the number of features (words/phrases) to 8000 most important ones. we can use any randome limited value here 
#Why? To avoid very large and slow models, especially with limited data.
#Without this, the model might try to use all possible words, which can be too many (100k+).
X = vectorizer.fit_transform(df_balanced["text"])
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the model ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print(" Model trained.")

# === Save the model and vectorizer ===
joblib.dump(clf, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f" Model saved to {MODEL_PATH}")
print(f" Vectorizer saved to {VECTORIZER_PATH}")

# === Evaluate the model ===
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
labels = sorted(df_balanced["label"].unique())
print(f" Accuracy: {accuracy:.2%}")

# ===  Class-wise metrics chart ===
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
plt.title(" Class-wise Metrics - NLP PROJECT")
plt.grid(axis='y', linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "classwise_metrics.png"))
plt.show()

# ===  Accuracy chart ===
plt.figure(figsize=(4, 5))
plt.bar(["Accuracy"], [accuracy], color="purple")
plt.ylim(0, 1)
plt.title(" Overall Accuracy - NLP PROJECT")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "accuracy_bar.png"))
plt.show()

# ===  Confusion matrix ===
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(" Confusion Matrix - NLP PROJECT")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"))
plt.show()
