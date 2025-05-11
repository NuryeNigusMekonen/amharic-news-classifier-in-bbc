
# Amharic News Classifier - BBC Dataset
```
This project is a machine learning-based text classifier that categorizes Amharic news articles into four distinct categories: **Health**, **Politics**, **Sport**, and **Technology**. The classifier is trained using an enhanced dataset sourced from BBC Amharic news, with 250 labeled articles per category.
```
## 📊 Project Highlights
```
- **Language**: Amharic 🇪🇹
- **Dataset**: 1000 labeled news articles (250 per category)
- **Model**: Logistic Regression (with TF-IDF vectorization)
- **Performance**: High precision, recall, and F1-score across all categories
```
## 🚀 Features
```
- Classifies Amharic news articles into 4 categories.
- Enhanced with an expanded dataset for better generalization.
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
- Visual performance metrics included.

```

## 📁 Project Structure

```

amharic-news-classifier-in-bbc/
│
├── classifier.py                  # Main model training script
├── testing\_with\_sample\_data.py   # Script for testing model on sample data
├── visualize\_using\_graph.py      # Visualization of results
├── tfidf\_vectorizer\_300+.pkl     # Saved TF-IDF vectorizer
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── data/
│   └── bbc\_amharic\_dataset\_300+.csv  # Labeled dataset (1000 samples)
│
├── graphs/
│   ├── accuracy\_bar.png              # Overall accuracy chart
│   ├── classwise\_metrics.png        # Precision/Recall/F1 by class
│   └── confusion\_matrix.png         # Confusion matrix heatmap

```
---

## 📈 Performance Results

### ✅ Overall Accuracy

![Accuracy](plots/accuracy_bar.png)

### 📊 Class-wise Metrics

![Class Metrics](plots/classwise_metrics.png)

### 🧩 Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

```
## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/NuryeNigusMekonen/amharic-news-classifier-in-bbc.git
cd amharic-news-classifier-in-bbc
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### Train the Classifier:

```bash
python classifier.py
```

### Test on Sample Data:

```bash
python testing_with_sample_data.py
```

### Visualize Results:

```bash
python visualize_using_graph.py
```

---

## 🧠 Model Details

* **Vectorization**: TF-IDF
* **Classifier**: Logistic Regression
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

---

## ✍️ Author

* **Nurye Nigus Mekonen**

Feel free to ⭐ the repo or contribute by opening issues or pull requests!

---

## 📄 License

This project is licensed under the MIT License.

