import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to predict sentiment
def predict_sentiment(text, model):
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    probabilities = model.predict_proba([text])[0]

    # Determine sentiment based on highest probability
    max_prob_index = np.argmax(probabilities)  # Index of the max probability
    prediction = [-1, 0, 1][max_prob_index]  # Map index to sentiment
    sentiment = sentiment_map.get(prediction, 'Unknown')

    return sentiment, probabilities, prediction

# Streamlit UI
st.title("Sentiment Prediction App")
st.write("Enter a text to analyze its sentiment (Positive, Neutral, or Negative).")

# Load the model
model_path = 'sentiment_model.sav'  # Path to the saved model
model = load_model(model_path)

if model is not None:
    # Input text box
    user_input = st.text_area("Enter text:", "")

    # Predict button
    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            # Predict sentiment
            sentiment, probabilities, prediction = predict_sentiment(user_input, model)
            
            # Display predicted sentiment
            st.subheader(f"Predicted Sentiment: {sentiment}")
            
            # Display probabilities
            st.write("**Probabilities:**")
            st.write(f"- Negative: {probabilities[0]:.2f}")
            st.write(f"- Neutral: {probabilities[1]:.2f}")
            st.write(f"- Positive: {probabilities[2]:.2f}")

            # Visualisasi distribusi probabilitas
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Probability': probabilities
            })
            
            # Plot bar chart
            fig, ax = plt.subplots()
            ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=['red', 'blue', 'green'])
            ax.set_ylim(0, 1)  # Set batas sumbu Y dari 0 hingga 1
            ax.set_ylabel('Probability')
            ax.set_title('Sentiment Probability Distribution')
            st.pyplot(fig)
        else:
            st.error("Please enter some text to analyze.")
else:
    st.error("Model could not be loaded. Please check the model file.")
