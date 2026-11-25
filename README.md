# Pakistan House Price Prediction (Zameen.com)

Advanced regression project: data cleaning, feature engineering (geo features, target encoding),
4 tuned models, explainability, and a Streamlit web app with prediction range.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Download the Zameen.com Pakistan property dataset (`zameen-updated.csv`) from Kaggle and put it in `data/`.
Without it the project runs on a small FAKE sample (`python scripts/make_sample_data.py`) - only for testing.

## Run (in this order)
```bash
python -m src.preprocessing      # clean data + split          (Person A)
python -m src.eda                # graphs + insights           (Person A)
python -m src.train_models       # train 4 models, pick best   (Person B)
streamlit run app.py             # web app                     (Person B)
```
Use `python -m src.train_models --fast` for a quick test run.

## Structure
```
app.py                  Streamlit web app (3 tabs)
src/config.py           paths, feature lists
src/preprocessing.py    cleaning, feature engineering, split
src/eda.py              EDA graphs and insights
src/train_models.py     training, tuning, comparison, feature importance
src/predict.py          inference helper used by the app
PROJECT_GUIDE.md        full explanation, work split, report outline, viva questions
```
