#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Information Retrieval Preprocessing Project
Prepared for: Dr. Akbarpour

Steps:
    1. Normalization + tokenization
    2. Linguistic analysis with spaCy (tokenization, POS, lemmatization)
    3. Stop-word removal (NLTK stopword list)
    4. Evaluation (Precision / Recall) against a fixed gold standard

Note:
    The gold standard is used ONLY to compute Precision and Recall.
    It does NOT participate in any preprocessing or linguistic decisions.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords

import spacy


# ---------------------------------------------------------------------------
# Gold standard
# (fixed reference, only for evaluation)
# ---------------------------------------------------------------------------

GOLD_STANDARD: Dict[str, List[str]] = {
    "doc1": ["friend", "roman", "countryman", "lend", "ear"],
    "doc2": [
        "john",
        "new",
        "usa",
        "laptop",
        "state",
        "art",
        "device",
        "price",
        "high",
        "quality",
        "amaze",
    ],
    "doc3": [
        "authorize",
        "authorize",
        "project",
        "pend",
        "authorize",
        "payment",
        "regular",
    ],
}


# ---------------------------------------------------------------------------
# NLTK setup (only stopwords)
# ---------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    """Ensure required NLTK resources (stopwords) are available."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_PUNCT_PATTERN = re.compile(r"[.,!?;:()\"']")


def normalize_text(text: str) -> str:
    """
    Basic normalization:
        - Collapse dotted abbreviations (U.S.A. -> USA)
        - Remove possessive 's (John's -> John)
        - Replace hyphens with spaces (state-of-the-art -> state of the art)
        - Remove punctuation
        - Lowercase everything
    """
    # U.S.A. -> USA  (sequence of X.X.X.)
    text = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda m: m.group(0).replace(".", ""),
        text,
    )

    # Remove possessive 's / ’s
    text = re.sub(r"['’]s\b", "", text)

    # Hyphens to spaces
    text = text.replace("-", " ")

    # Remove punctuation
    text = _PUNCT_PATTERN.sub(" ", text)

    # Lowercase
    text = text.lower()

    return text


# ---------------------------------------------------------------------------
# spaCy-based preprocessing
# ---------------------------------------------------------------------------

def process_document_spacy(text: str, stop_words: set, nlp) -> List[str]:
    """
    Full preprocessing of a single document using spaCy:
        - normalization
        - spaCy tokenization, POS, lemmatization
        - stop-word removal (using NLTK stopword set)
    """
    normalized = normalize_text(text)
    doc = nlp(normalized)

    processed: List[str] = []

    for token in doc:
        # Skip spaces and punctuation
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.lower().strip()

        if not lemma:
            continue

        # Stop-word filtering (based on lemma and raw form)
        if lemma in stop_words or token.text in stop_words:
            continue

        processed.append(lemma)

    return processed


# ---------------------------------------------------------------------------
# Evaluation (NO stemming, gold is fixed)
# ---------------------------------------------------------------------------

def precision_recall(predicted: List[str], gold: List[str]) -> Tuple[float, float]:
    """
    Compute precision and recall by directly comparing
    the predicted tokens to the (fixed) gold standard tokens.

    Gold is NOT modified in any way.
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    if not pred_set and not gold_set:
        return 1.0, 1.0

    true_pos = len(pred_set & gold_set)

    precision = true_pos / len(pred_set) if pred_set else 0.0
    recall = true_pos / len(gold_set) if gold_set else 0.0

    return precision, recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_nltk_data()

    # Load NLTK stopwords
    eng_stopwords = set(stopwords.words("english"))

    # Load spaCy English model
    # Make sure you have run:
    #   python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    print("=== IR Preprocessing Pipeline (spaCy Lemmatization) ===")
    print("Enter number of documents:")
    try:
        n_docs = int(input("> ").strip())
    except ValueError:
        print("Invalid number. Exiting.")
        return

    raw_docs: Dict[str, str] = {}
    for i in range(1, n_docs + 1):
        doc_id = f"doc{i}"
        print(f"\nEnter text for {doc_id}:")
        text = input().strip()
        raw_docs[doc_id] = text

    outputs: Dict[str, List[str]] = {}
    eval_results: Dict[str, Dict[str, float]] = {}

    print("\n=== Processed Outputs ===")
    for doc_id, text in raw_docs.items():
        processed = process_document_spacy(text, eng_stopwords, nlp)
        outputs[doc_id] = processed

        print(f"\n{doc_id}:")
        print("Tokens:", processed)

        gold_tokens = GOLD_STANDARD.get(doc_id)
        if gold_tokens is not None:
            p, r = precision_recall(processed, gold_tokens)
            eval_results[doc_id] = {"precision": p, "recall": r}
            print(f"Precision({doc_id}): {p:.3f}")
            print(f"Recall   ({doc_id}): {r:.3f}")
        else:
            print(f"(No gold standard defined for {doc_id}.)")

    # Write outputs to JSON
    with open("outputs.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    if eval_results:
        with open("evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print("\nAll processed outputs written to 'outputs.json'.")
    if eval_results:
        print("Evaluation metrics written to 'evaluation.json'.")


if __name__ == "__main__":
    main()
