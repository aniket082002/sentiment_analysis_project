from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for reviews in a CSV file.')
    parser.add_argument('--input', type=Path, default=ROOT / 'reviews.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'predicted_reviews.csv')
    args = parser.parse_args()
    model_path = ROOT / 'models' / 'sentiment_model.pkl'
    if not model_path.exists():
        parser.error('Trained model missing. Run python train_model.py first.')
    if not args.input.exists():
        parser.error(f'Input file missing: {args.input}')
    df = pd.read_csv(args.input)
    if 'review' not in df.columns:
        parser.error('Input CSV must have a review column.')
    model = joblib.load(model_path)
    if df.empty:
        df['predicted_sentiment'] = pd.Series(dtype='str')
        df['confidence'] = pd.Series(dtype='float')
    else:
        reviews = df['review'].fillna('').astype(str)
        valid = reviews.str.strip().ne('')
        df['predicted_sentiment'] = ''
        df['confidence'] = float('nan')
        if valid.any():
            df.loc[valid, 'predicted_sentiment'] = model.predict(reviews[valid])
            df.loc[valid, 'confidence'] = model.predict_proba(reviews[valid]).max(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Saved {len(df)} rows to {args.output}')

if __name__ == '__main__':
    main()
