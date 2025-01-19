import pickle
import numpy as np

# Load the trained model
try:
    with open("naive_bayes_model.sav", "rb") as model_file:
        classifier_nb = pickle.load(model_file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please ensure 'naive_bayes_model.sav' exists in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to predict sentiment
def predict_sentiment(text, model):
    # Map sentiment labels
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

    try:
        # Predict probabilities
        probabilities = model.predict_proba([text])[0]
        max_prob_index = np.argmax(probabilities)  # Index of max probability
        prediction = [-1, 0, 1][max_prob_index]  # Map index to sentiment
        sentiment = sentiment_map.get(prediction, 'Unknown')

        return sentiment, probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", [0, 0, 0]  # Return default probabilities on error

# Interactive CLI
def main():
    print("\n--- Sentiment Prediction ---")
    while True:
        user_input = input("Enter text to analyze sentiment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        # Validate input
        if not user_input.strip():
            print("Please enter valid text.\n")
            continue

        # Predict sentiment
        sentiment, probabilities = predict_sentiment(user_input, classifier_nb)
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Probabilities: Negative={probabilities[0]:.2f}, Neutral={probabilities[1]:.2f}, Positive={probabilities[2]:.2f}\n")

if __name__ == "__main__":
    main()
