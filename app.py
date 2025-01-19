import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to load evaluation data
def load_evaluation_data(data_path):
    try:
        data = joblib.load(data_path)
        return data
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
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

# Visualisasi distribusi probabilitas
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment, probabilities, prediction = predict_sentiment(user_input, model)
        st.subheader(f"Predicted Sentiment: {sentiment}")
        
        # Tampilkan grafik distribusi probabilitas
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Sentiment'))

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Streamlit UI
st.title("Sentiment Prediction App")
st.write("Enter a text to analyze its sentiment (Positive, Neutral, or Negative).")

# Load the model
model_path = 'sentiment_model.sav'  # Path to the saved model
model = load_model(model_path)

# Load evaluation data
evaluation_data_path = 'evaluation_data.sav'  # Path to the saved evaluation data
evaluation_data = load_evaluation_data(evaluation_data_path)

if model is not None and evaluation_data is not None:
    # Input text box
    user_input = st.text_area("Enter text:", "")

    # Predict button
    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            sentiment, probabilities, prediction = predict_sentiment(user_input, model)
            st.subheader(f"Predicted Sentiment: {sentiment}")
            st.write("**Probabilities:**")
            st.write(f"- Negative: {probabilities[0]:.2f}")
            st.write(f"- Neutral: {probabilities[1]:.2f}")
            st.write(f"- Positive: {probabilities[2]:.2f}")
        else:
            st.error("Please enter some text to analyze.")

    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    st.write("Below are the evaluation metrics for the trained model:")

    # Extract evaluation data
    y_true = evaluation_data['y_true']
    y_pred = evaluation_data['y_pred']
    y_prob = evaluation_data['y_prob']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Recall", f"{recall:.2f}")
    col3.metric("Precision", f"{precision:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")

    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred)

    # Plot ROC curve
    st.subheader("ROC Curve")
    plot_roc_curve(y_true, y_prob)

else:
    st.error("Model or evaluation data could not be loaded. Please check the files.")
