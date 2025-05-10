# 📰 Amharic News Classifier 

This project uses Logistic Regression to classify Amharic news headlines into one of four categories:  
**Politics**, **Health**, **Sport**, or **Technology**.

It demonstrates an end-to-end machine learning workflow in a low-resource language using real-world data scraped from **BBC Amharic**.


Features

-  Balanced Amharic news dataset (400+ headlines)
-  Text preprocessing and normalization
-  TF-IDF vectorization with bigrams
-  Logistic Regression model
-  Precision, Recall, F1-score, and Confusion Matrix
-  Graphical evaluation with Matplotlib and Seaborn
-  Model and vectorizer saving with `joblib`

---
'''
## 📁 Folder Structure

```
amharic-news-classifier/
│
├── data/
│   └── bbc_amharic_dataset_300+.csv
│
├── models/
│   ├── amharic_news_classifier_300+.pkl
│   └── tfidf_vectorizer_300+.pkl
│
├── scripts/
│   ├── classifier.py
│   ├── testing_with_sample_data.py
│   ├── import_requests_for_for_mother.py
│   └── visualize_using_graph.py
│
├── figures/
│   ├── accuracy_score_overal.png
│   ├── confusion_matrix_amharic.png
│   └── Figure_1.png
│
├── docs/
│   └── NLP_Projects_Guide.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

Sample Evaluation

Overall Accuracy: `56.67%`

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Health      | 0.26      | 0.73   | 0.38     |
| Politics    | 0.88      | 0.54   | 0.67     |
| Sport       | 1.00      | 0.62   | 0.76     |
| Technology  | 0.64      | 0.43   | 0.51     |



## Confusion Matrix
```

             Predicted
            ┌──────────------------------------┐
Actual      │ Health Poletics Sport Technology │
┌───        ┼───────--------------------───────┤
Health      │11      1        0     3          │
Poletics    │11      15       0     2          │
Sport       │10      0        16    0          │
Technology  │11      1        0     9          │
            ------------------------------------
			
```


 Installation & Usage

```bash
# 1. Clone this repository
git clone https://github.com/your-username/amharic-news-classifier.git
cd amharic-news-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Visualize results
python visualise_using_graph.py



---
Future Improvements
Collect more diverse and balanced data

Integrate Amharic stopwords and stemming

Replace TF-IDF with pretrained embeddings (e.g., fastText, BERT)

Explore deep learning models (LSTM, transformer)

👤 Author
Nurye Nigus
Electrical & Software Engineer
📧 nurye.nigus.me@gmail.com
🌐 LinkedIn (https://www.linkedin.com/in/nryngs/)
🐙 GitHub: @NuryeNigusMekonen


