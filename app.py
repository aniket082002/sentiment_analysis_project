from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'sentiment_model.pkl'
st.set_page_config(page_title='Sentiment Analyzer', page_icon='💬')
st.title('💬 Customer Review Sentiment Analyzer')
st.write('Classify an English review as positive or negative.')
if not MODEL.exists():
    st.error('No trained model found. Run `python train_model.py` first (see README).')
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

model = load_model()
review = st.text_area('Enter a customer review', height=130, placeholder='This product is amazing!')
if st.button('Predict Sentiment', type='primary'):
    if not review.strip():
        st.warning('Please enter a review.')
    else:
        sentiment = model.predict([review])[0]
        confidence = max(model.predict_proba([review])[0])
        if sentiment == 'positive':
            st.success(f'😊 Positive — estimated confidence {confidence:.1%}')
        else:
            st.error(f'😞 Negative — estimated confidence {confidence:.1%}')
        st.caption('Model confidence is not a guarantee of correctness.')
