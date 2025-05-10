import joblib

# Load saved model and vectorizer
clf = joblib.load("amharic_news_classifier_300+.pkl")
vectorizer = joblib.load("tfidf_vectorizer_300+.pkl")

# Add new sample headline(s)
new_texts = [
    "አዲስ ቴክኖሎጂ በኢትዮጵያ የተመረቀ ነው",
    "የአሜሪካ በቅርቡ ያባረረቻቸው የደቡብ አፍሪካው አምባሳደር ማን ናቸው?"
]
# Transform and predict
new_vec = vectorizer.transform(new_texts)
predictions = clf.predict(new_vec)

# Show results
for text, pred in zip(new_texts, predictions):
    print(f"\n \"{text}\"\n Predicted Category: {pred}")
