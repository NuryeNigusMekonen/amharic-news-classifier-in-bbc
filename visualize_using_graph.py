import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import joblib
# Load model and vectorizer
clf = joblib.load("amharic_news_classifier_300+.pkl")
vectorizer = joblib.load("tfidf_vectorizer_300+.pkl")

# Load our balanced dataset and re-split
df = pd.read_csv("bbc_amharic_dataset_300+.csv")
df.dropna(subset=["text", "label"], inplace=True)
# Clean text
import re
df["text"] = df["text"].apply(lambda x: re.sub(r"[፣።፤፥,!?\"']", "", str(x)).strip())

# Balance again
min_class = df['label'].value_counts().min()
df_balanced = pd.concat([
    df[df['label'] == label].sample(min_class, random_state=42)
    for label in df['label'].unique()
])

# Vectorize again
X = vectorizer.transform(df_balanced['text'])
y = df_balanced['label']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and get classification report
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Predict and get classification report
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

#  Calculate and print overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Overall Accuracy: {accuracy:.2%}")


# Prepare data
labels = list(df_balanced['label'].unique())
metrics = ['precision', 'recall', 'f1-score']

# Extract values for each label
data = {metric: [report[label][metric] for label in labels] for metric in metrics}
x = np.arange(len(labels))  # label locations
width = 0.25

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, data['precision'], width, label='Precision', color='steelblue')
rects2 = ax.bar(x, data['recall'], width, label='Recall', color='orange')
rects3 = ax.bar(x + width, data['f1-score'], width, label='F1-Score', color='green')

# Labels and titles 
ax.set_ylabel('Score')
ax.set_title(' Class-wise Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
#  Plot overall accuracy as a separate chart
plt.figure(figsize=(4, 5))
plt.bar(["Accuracy"], [accuracy], color="purple")
plt.ylim(0, 1)
plt.title(" Overall Accuracy")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.show()
#confustion matrix
# Generate confusion matrix
labels = sorted(df_balanced["label"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(" Confusion Matrix - Amharic Text Classification")
plt.tight_layout()
plt.savefig("confusion_matrix_amharic.png")  # Saves image in our working directory 
plt.show()