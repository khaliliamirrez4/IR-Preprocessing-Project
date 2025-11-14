 IR Preprocessing Project

This repository contains the implementation of a text preprocessing pipeline for the **Information Retrieval** course.

🔹 Features

- Text normalization (punctuation removal, case folding, possessive removal, hyphen handling)
- Tokenization
- POS tagging (NLTK)
- Lemmatization (WordNet Lemmatizer)
- Stopword removal (NLTK English stopword list)
- Evaluation using **Precision** and **Recall** against a fixed **Gold Standard**

> Note: The Gold Standard and the corresponding evaluation are defined for three default documents inside the code.  
> If you want to evaluate other texts, you must manually update the Gold Standard in the source code.

---

📁 Project Structure

```text
IR-Preprocessing-Project/
│
├── IR.py          # Main preprocessing and evaluation script
├── outputs.json   # Processed tokens per document (generated after running)
├── evaluation.json# Precision/Recall results (generated after running)
└── README.md      # Project description and usage
▶️ How to Run
1. Install dependencies
pip install nltk
Then, in a Python shell (only once):
import nltk
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
2. Run the script
python IR.py
You will be asked:

To enter the number of documents

To input the text for each document

The script will:

Preprocess each document

Print the resulting tokens

Compute Precision and Recall for the three default documents that have a Gold Standard defined in the code

Save results to outputs.json and evaluation.json
🔧 Modifying the Gold Standard

Inside IR.py, there is a dictionary similar to:
GOLD_STANDARD = {
    "doc1": [...],
    "doc2": [...],
    "doc3": [...],
}
If you want to evaluate different texts or change the target tokens:

Edit these lists according to your new documents

Re-run the script to compute new Precision/Recall values
👤 Author

Amirreza Khalili - student of Computer Engineering
