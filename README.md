# German Email Spam Detection & Clustering

A Streamlit app that classifies German emails as spam or not spam and groups them with unsupervised clustering. The pipeline is TF-IDF, then TruncatedSVD, then KMeans clustering, with an XGBoost spam classifier on top.

## Project files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Streamlit app (entry point) |
| `German_Email_Analysis_Corrected.ipynb` | Analysis and model training notebook |
| `german_email.csv`, `german_email_labeled.csv`, `german_spam_data.csv` | Datasets |
| `tfidf_vectorizer.pkl`, `svd_reducer.pkl`, `kmeans_clusterer.pkl`, `spam_clusters.pkl`, `xgboost_spam_classifier.pkl` | Trained models loaded by the app |

## Python version

The app was developed and tested with **Python 3.13** (3.13.9).

Pinned package versions (the `.pkl` models were saved with these, so use the same versions to avoid unpickling errors):

| Package | Version |
|---------|---------|
| streamlit | 1.52.2 |
| scikit-learn | 1.8.0 |
| xgboost | 3.1.2 |
| pandas | 2.3.3 |
| numpy | 2.4.0 |
| plotly | 6.5.0 |
| joblib | 1.5.3 |
| spacy | 3.8.11 |
| de_core_news_sm | 3.8.0 |

## Run locally

```bash
python -m venv appenv
appenv\Scripts\activate          # Windows
# source appenv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python -m spacy download de_core_news_sm
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (all `.pkl` and `.csv` files must be committed, since the app loads them from the repo root).
2. Make sure the repo root contains a `requirements.txt` with:

   ```text
   streamlit==1.52.2
   scikit-learn==1.8.0
   xgboost==3.1.2
   pandas==2.3.3
   numpy==2.4.0
   plotly==6.5.0
   joblib==1.5.3
   spacy==3.8.11
   de_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl
   ```

   The last line installs the German spaCy model, which the app needs at startup.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **Create app**, then choose this repository, the `main` branch, and `streamlit_app.py` as the main file.
5. Open **Advanced settings** and select **Python 3.13** (match the version above).
6. Click **Deploy**. The first build takes a few minutes while dependencies install.

### Troubleshooting

- **`German spaCy model not found`**: the `de_core_news_sm` line is missing from `requirements.txt`.
- **Unpickling or version errors**: the package versions differ from the ones the models were saved with. Use the pinned versions above.
- **File not found for a `.pkl` or `.csv`**: the file was not committed, or is not in the repo root.
