# Import necessary libraries
import pydot
import graphviz
from tensorflow.keras.utils import plot_model
import streamlit as st
import base64
from PIL import Image
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO
import pandas as pd
from contextlib import redirect_stdout

st.set_page_config(page_title='Sentiment Analysis', page_icon='😃', layout='wide', initial_sidebar_state='auto')
# Load max_seq_length from file
max_seq_length_path = './data/glove/max_seq_length.txt'
with open(max_seq_length_path, 'r') as f:
    max_seq_length = int(f.read())

# Function to remove URLs from text
def remove_urls(text):
    if not isinstance(text, str):
        return text
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, '', text)

# Function to preprocess new input data
def preprocess_input(text, tokenizer, max_seq_length):
    text = [text]
    text = remove_urls(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    return padded_sequences

# Load a pre-trained tokenizer
tokenizer = Tokenizer()
tokenizer.word_index = np.load('./data/glove/tokenizer_word_index.npy', allow_pickle=True).item()

# Function to classify a tweet using a loaded model
def classify_tweet_with_model(tweet, model):
    new_input_text = tweet
    preprocessed_input = preprocess_input(new_input_text, tokenizer, max_seq_length)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ["negative", "neutral", "positive"]
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Define paths to pre-trained models
model_paths = {
    "Model Flatten": {
        "model_path": './models/glove_trained/best_model_flatten.h5'
    },
    "Model CONV": {
        "model_path": './models/glove_trained/best_model_conv.h5'
    },
    "Model GRU": {
        "model_path": './models/glove_trained/best_model_gru.h5'
    },
    "Model LSTM": {
        "model_path": './models/glove_trained/best_model_lstm.h5'
    },
    "Model LSTM2": {
        "model_path": './models/glove_trained/best_model_lstm2.h5'
    },
    "Model LSTM CONV": {
        "model_path": './models/glove_trained/best_model_lstm_con.h5'
    },
    "Model ATTENTION": {
        "model_path": './models/glove_trained/best_model_with_attention.h5'
    }
}

# Create a Streamlit app
def main():
    # Add a banner image
    banner_image = Image.open("./images/sentimentanalysishotelgeneric-2048x803-1.jpg")
    st.image(banner_image, use_column_width=True)

    st.title("Sentiment Analysis with Pre-trained Model")

    # User input for text
    user_input = st.text_area("Enter a text:", "I love this product!", key="text_input")

    # Create a dropdown for model selection with a unique key
    model_selection = st.selectbox("Select a Model", list(model_paths.keys()), key="model_selection")

    # Load the selected model and associated data
    selected_model = model_paths[model_selection]

    # Load the pre-trained model
    loaded_model = tf.keras.models.load_model(selected_model['model_path'])

    if st.button("Classify Sentiment"):

        # Preprocess input for the selected model
        preprocessed_input = preprocess_input(user_input, tokenizer, max_seq_length)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(preprocessed_input)

        # Map the predicted class to your class labels
        class_labels = ["negative", "neutral", "positive"]
        predicted_class = class_labels[predictions.argmax(axis=1)[0]]

        # Display the predicted sentiment
        st.markdown(f'**Predicted Sentiment:** {predicted_class.capitalize()}')

        # Display the model architecture using plot_model within an expander
        with st.expander("Click to view Model Architecture"):
            st.text("This is a visual representation of the model's architecture:")

            # Create a temporary file to save the model architecture plot
            model_plot_path = "model_plot.png"

            # Save the model architecture plot as PNG
            tf.keras.utils.plot_model(loaded_model, to_file=model_plot_path, show_shapes=True, show_layer_names=True,
                                      dpi=96)

            # Display the saved model architecture plot
            st.image(model_plot_path)

        # Display the model summary as text
        with st.expander("Click to view Model Summary"):
            model_summary = ""
            # Capture the model summary into a string
            try:
                # Create a temporary file to capture stdout
                with open('model_summary.txt', 'w') as f:
                    # Use the context manager to redirect stdout
                    with redirect_stdout(f):
                        loaded_model.summary()
                # Read the captured model summary back into the string
                with open('model_summary.txt', 'r') as f:
                    model_summary = f.read()
            except Exception as e:
                model_summary = f"Error capturing model summary: {str(e)}"
            st.text(model_summary)

        # Display the probabilities of each class in a table
        st.subheader("Class Probabilities")
        probability_df = pd.DataFrame({"Class": class_labels, "Probability": predictions[0]})
        st.table(probability_df)

        # Display an image based on the sentiment
        if predicted_class == "positive":
            sentiment_image = Image.open("./images/positive.jpg")
        elif predicted_class == "negative":
            sentiment_image = Image.open("./images/negative.jpg")
        else:
            sentiment_image = Image.open("./images/neutral.jpg")

        # Resize the image for display
        image_size = (100, 100)
        #resized_image = sentiment_image.resize(image_size, Image.ANTIALIAS)
        resized_image = sentiment_image.resize(image_size, Image.BILINEAR)


        # Convert the image to base64
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display resized image with markdown spacer for center alignment
        st.markdown(
            f'<p align="center"><img src="data:image/png;base64,{img_str}" alt="{predicted_class}" width="{image_size[0]}"></p>',
            unsafe_allow_html=True
        )

# Run the Streamlit app if the script is executed
if __name__ == "__main__":
    main()
