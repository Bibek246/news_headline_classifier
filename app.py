import os
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.nn.functional import softmax

# Must be the very first Streamlit command
st.set_page_config(page_title="News Topic Classifier", layout="wide")

# --- Categories ---
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

# --- Load models once ---
@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf.pkl")
    lr    = joblib.load("models/logreg.pkl")
    mlp_path = "models/mlp.pt"
    mlp   = torch.load(mlp_path, map_location="cpu") if os.path.exists(mlp_path) else None
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    bert_model     = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    bert_model.eval()
    return tfidf, lr, mlp, bert_tokenizer, bert_model

tfidf, lr_model, mlp_model, bert_tokenizer, bert_model = load_models()

# --- Prediction helpers ---
def predict_lr(text):
    vec = tfidf.transform([text])
    pred = lr_model.predict(vec)[0]
    probs = lr_model.predict_proba(vec)[0]
    return pred, probs

def predict_mlp(text):
    if mlp_model is None:
        return None, None
    vec = tfidf.transform([text]).toarray()
    x = torch.from_numpy(vec).float()
    with torch.no_grad():
        logits = mlp_model(x).squeeze()
        probs = softmax(logits, dim=0).numpy()
        pred = int(np.argmax(probs))
    return pred, probs

def predict_bert(text):
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits.squeeze()
        probs = softmax(logits, dim=0).numpy()
        pred = int(np.argmax(probs))
    return pred, probs

# --- Streamlit app UI ---
st.markdown(
    "<h1 style='text-align: center; color: #333;'>News Headline Topic Classifier</h1>",
    unsafe_allow_html=True
)
st.sidebar.header("Options")

example = st.sidebar.selectbox("Try an example headline:", [
    "",
    "Global Leaders Gather in Paris for Climate Summit",
    "Lakers Edge Warriors 112–108 in Overtime Thriller",
    "Amazon Announces Prime Day Will Expand to Two Weeks",
    "Google Unveils New AI That Translates 50 Languages"
])

model_choice = st.sidebar.multiselect(
    "Select model(s):",
    ["Logistic Regression", "MLP", "DistilBERT"],
    default=["DistilBERT"]
)

headline = st.text_input(" Enter a news headline:", value=example)

if st.button("Classify"):
    if not headline.strip():
        st.error("Please enter or select a headline.")
    else:
        results = []
        if "Logistic Regression" in model_choice:
            results.append(("Logistic Regression", *predict_lr(headline)))
        if "MLP" in model_choice:
            mlp_pred, mlp_probs = predict_mlp(headline)
            if mlp_pred is not None:
                results.append(("MLP", mlp_pred, mlp_probs))
        if "DistilBERT" in model_choice:
            results.append(("DistilBERT", *predict_bert(headline)))

        cols = st.columns(len(results))
        for (name, pred, probs), col in zip(results, cols):
            col.markdown(f"### {name}")
            col.success(f"**{CATEGORIES[pred]}**")
            df = pd.DataFrame({
                "Category": CATEGORIES,
                "Confidence": np.round(probs, 3)
            }).sort_values("Confidence", ascending=False)
            col.table(df)

        st.balloons()
