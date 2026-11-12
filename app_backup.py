import streamlit as st
import pickle

# Load the trained sentiment analysis model
with open('sentiment_analysis.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment.')

review = st.text_area('Enter your movie review:')

if st.button('Predict Sentiment'):
    if review.strip():
        prediction = model.predict([review])

        if prediction[0] == 'positive':
            st.success('Positive Review 😊')
        else:
            st.error('Negative Review 😞')
    else:
        st.warning('Please enter a movie review first.')