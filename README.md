#  Plagiarism Checker using NLP

A simple NLP-powered Streamlit app to check semantic similarity between two pieces of text using Sentence Transformers.

## 🚀 Features

- Compare two texts for semantic similarity
- Detect possible plagiarism using cosine similarity
- Easy-to-use Streamlit UI

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 📊
- Sentence Transformers 🤖 (all-MiniLM-L6-v2)
- PyTorch backend

## 📦 Installation

```bash
git clone https://github.com/anasnorani1/plagiarism_checker.git
cd plagiarism_checker
pip install -r requirements.txt
```
## 📄 Usage
streamlit run app.py
## ✨ Example
- Paste two texts and click Check Similarity to get a semantic similarity score with interpretation.

## 📚 Model Used
- all-MiniLM-L6-v2 from SentenceTransformers
