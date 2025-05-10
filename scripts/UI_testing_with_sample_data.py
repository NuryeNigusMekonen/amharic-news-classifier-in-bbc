import tkinter as tk
from tkinter import messagebox
import joblib
import re
import os

# lets choose our directory 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
clf = joblib.load(os.path.join(MODEL_DIR, "amharic_news_classifier_300+.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_300+.pkl"))

# Clean text input
def clean_text(text):
    return re.sub(r"[፣።፤፥,!?\"']", "", text.strip())

# Predict category
def predict_category():
    input_text = entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Input Error", " Please enter an Amharic headline first.")
        return
    cleaned = clean_text(input_text)
    vector = vectorizer.transform([cleaned])
    prediction = clf.predict(vector)[0]
    result_label.config(text=f" Predicted Category: {prediction}", fg="blue")

# UI Window Setup
root = tk.Tk()
root.title("Amharic News Headline Classifier - HILCOE")
root.geometry("550x350")
root.resizable(False, False)
root.configure(bg="#f5f5f5")

# Fonts and styles
font_main = ("Segoe UI", 12)
font_result = ("Segoe UI", 14, "bold")

# Layout
tk.Label(root, text=" Enter an Amharic News Headline Below:", font=font_main, bg="#f5f5f5").pack(pady=10)
entry = tk.Text(root, height=5, width=60, font=font_main, wrap="word")
entry.pack(pady=5)

tk.Button(root, text=" Predict", command=predict_category, bg="#007acc", fg="white", font=font_main).pack(pady=10)

result_label = tk.Label(root, text="", font=font_result, bg="#f5f5f5")
result_label.pack(pady=5)

root.mainloop()
