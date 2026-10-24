from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='Train a sentiment classifier on a labeled CSV.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'IMDB Dataset.csv')
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f'Dataset not found: {args.data}. See README.md for download instructions.')
    df = pd.read_csv(args.data)
    if not {'review', 'sentiment'}.issubset(df.columns):
        parser.error('Dataset needs review and sentiment columns.')
    df = df.dropna(subset=['review', 'sentiment']).copy()
    df['review'] = df['review'].astype(str).str.strip()
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    df = df[(df['review'] != '') & df['sentiment'].isin(['positive', 'negative'])]
    if len(df) < 10 or df['sentiment'].value_counts().min() < 2:
        parser.error('Need at least 10 valid reviews and at least 2 of each sentiment.')
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=.2, random_state=42, stratify=df['sentiment'])
    model = Pipeline([('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')), ('classifier', LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predicted):.4f}')
    print(classification_report(y_test, predicted, zero_division=0))
    path = ROOT / 'models' / 'sentiment_model.pkl'
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f'Model saved: {path}')

if __name__ == '__main__':
    main()
