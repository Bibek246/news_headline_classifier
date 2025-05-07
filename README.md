# News Headline Topic Classifier

This repository contains a full pipeline for classifying news headlines into one of four categories:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Three modeling approaches are implemented and compared:

1. **TF-IDF + Logistic Regression (Baseline)**
2. **TF-IDF + MLP**
3. **DistilBERT Fine-Tuning**

All experiment results, confusion matrices, and comparison charts are saved in the `outputs/` folder.

---

## Repository Structure

├── models/ # Saved model artifacts (TF-IDF, LR, MLP, BERT)
├── outputs/ # Generated plots, CSV logs, and final comparison
│ ├── baseline_confmat_val.png
│ ├── mlp_confmat_val.png
│ ├── bert_confmat_val.png
│ ├── baseline_preds_val.csv
│ ├── mlp_preds_val.csv
│ ├── bert_preds_val.csv
│ ├── baseline_preds_test.csv
│ ├── bert_preds_test.csv
│ ├── results.csv
│ └── model_comparison.png
├── news_headline_classifier.py # Main training & evaluation script
├── app.py # Streamlit demo web app
├── requirements.txt # Python dependencies
└── README.md # This file


---

## Setup & Dependencies

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd <repo_folder>
**Create a virtual environment**

python3 -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

**Install dependencies**
pip install -r requirements.txt

Key libraries: scikit-learn, torch, transformers, datasets, pandas, seaborn, streamlit

Running the Training & Evaluation Pipeline
Execute the main script to train all models, evaluate on validation and test splits, and save outputs:
python news_headline_classifier.py

This will:

Subsample 50 k headlines for training

Train TF-IDF + Logistic Regression, TF-IDF + MLP, and fine-tune DistilBERT

Generate confusion matrices (validation) and save as PNGs

Save prediction logs (true vs predicted) as CSVs

Evaluate baseline and BERT on test set

Save final accuracy comparison chart and results.csv

**Launching the Demo Web App**

**To interactively classify new headlines:**

streamlit run app.py

Choose between Logistic Regression, MLP, and DistilBERT

Try example headlines or enter your own

View prediction and confidence scores

**License**
This project is released under the MIT License. Feel free to use and modify!
