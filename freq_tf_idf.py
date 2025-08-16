import webbrowser
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def process_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        documents = file.readlines()

    
    documents = [line.strip() for line in documents if line.strip()]

    if not documents:
        return [], []

    
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(documents)
    freq_scores = np.array(count_matrix.sum(axis=0)).flatten()
    terms = count_vectorizer.get_feature_names_out()
    freq_dict = dict(zip(terms, freq_scores))
    top_freq_terms = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_scores))
    top_tfidf_terms = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    return top_freq_terms, top_tfidf_terms

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        
        top_freq_terms, top_tfidf_terms = process_text(file_path)

        return render_template("index.html", top_freq_terms=top_freq_terms, top_tfidf_terms=top_tfidf_terms)

    return render_template("index.html")

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")  
    app.run(debug=True, host="127.0.0.1", port=5000)
