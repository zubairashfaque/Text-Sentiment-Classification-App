# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import re
import plotly.io as pio

# Define a function to remove URLs from text
def remove_urls(text):
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, '', text)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, train_data, train_labels, val_data, val_labels, test_data, test_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(
        f'../models/glove_trained/best_{model.name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    history = model.fit(train_data, train_labels, epochs=100, batch_size=128,
                        validation_data=(val_data, val_labels), verbose=1,
                        callbacks=[early_stopping, checkpoint_callback])
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    return history, test_loss, test_accuracy

# Define the path to the CSV file
csv_file_path = '../data/train/train.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Drop the 'selected_text' column
df = df.drop(columns=['selected_text'])

# Convert the 'text' column to string type
df["text"] = df["text"].astype(str)

# Split the data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Remove URLs from text in the dataframes
train_df['text'] = train_df['text'].apply(remove_urls)
val_df['text'] = val_df['text'].apply(remove_urls)
test_df['text'] = test_df['text'].apply(remove_urls)

# Initialize a Tokenizer
tokenizer = Tokenizer()

# Fit the Tokenizer on the training data text
tokenizer.fit_on_texts(train_df['text'])

# Define the embedding dimension
embedding_dim = 100

# Initialize an embedding index dictionary
embedding_index = {}

# Load pre-trained word embeddings (GloVe)
with open('../data/glove/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Reinitialize the Tokenizer to account for word limit based on the pre-trained embeddings
tokenizer = Tokenizer(num_words=len(embedding_index) + 1)
tokenizer.fit_on_texts(train_df['text'])

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Create an embedding matrix for the words in the vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert text sequences to numerical sequences
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
val_sequences = tokenizer.texts_to_sequences(val_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Find the maximum sequence length
max_seq_length = max(max(len(seq) for seq in train_sequences),
                     max(len(seq) for seq in val_sequences),
                     max(len(seq) for seq in test_sequences))

# Pad sequences to have the same length
train_data = pad_sequences(train_sequences, maxlen=max_seq_length, padding='post')
val_data = pad_sequences(val_sequences, maxlen=max_seq_length, padding='post')
test_data = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post')

# Define a mapping of sentiment labels to numerical values
label_mapping = {"positive": 2, "neutral": 1, "negative": 0}

# Convert sentiment labels to numerical values
train_labels = np.array([label_mapping[label] for label in train_df['sentiment']])
val_labels = np.array([label_mapping[label] for label in val_df['sentiment']])
test_labels = np.array([label_mapping[label] for label in test_df['sentiment']])


# Save the tokenizer and max_seq_length for later use
np.save('../data/glove/tokenizer_word_index.npy', tokenizer.word_index)
with open('../data/glove/max_seq_length.txt', 'w') as f:
    f.write(str(max_seq_length))

# List of models with custom names
models = [
    (tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_gru'), 'model_gru'),

    (tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_lstm'), 'model_lstm'),

    (tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_lstm2'), 'model_lstm2'),

    (tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_conv'), 'model_conv'),

    (tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_flatten'), 'model_flatten'),

    (tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), name='input_layer'),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_lstm_con'), 'model_lstm_con'),

    (tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), name='input_layer'),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_seq_length, trainable=False),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='model_with_attention'), 'model_with_attention')
]

# Define a function to display the model summary and architecture
def display_model_summary(model):
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# Create a list to store results (history, test_loss, test_accuracy) for each model
results = []

# Loop through the models and train/evaluate them
for model, model_name in models:
    print(f"Displaying summary and architecture of {model_name}:")
    display_model_summary(model)

    # Save the model architecture as an image
    architecture_image_path = f'../models/glove_trained/{model_name}_architecture.png'
    plot_model(model, to_file=architecture_image_path, show_shapes=True, show_layer_names=True)
    print(f"Saved architecture diagram as: {architecture_image_path}")

    print(f"Training and evaluating {model_name}...")

    # Train and evaluate the model
    history, test_loss, test_accuracy = train_and_evaluate_model(model, train_data, train_labels, val_data,
                                                                 val_labels, test_data, test_labels)

    # Store results for this model
    results.append({
        "model_name": model_name,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    # Plot individual training and validation accuracy using Plotly
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(
        go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)), y=history.history['accuracy'],
                   mode='lines+markers', name='Train Accuracy'))
    accuracy_fig.add_trace(
        go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)), y=history.history['val_accuracy'],
                   mode='lines+markers', name='Validation Accuracy'))
    accuracy_fig.update_layout(title=f'{model_name} - Training and Validation Accuracy', xaxis_title='Epoch',
                               yaxis_title='Accuracy')
    # Save the accuracy graph as an image
    accuracy_image_path = f'../models/glove_trained/{model_name}_accuracy.png'
    pio.write_image(accuracy_fig, accuracy_image_path, format='png')

    # Show the accuracy graph
    accuracy_fig.show()

    # Plot individual training and validation loss using Plotly
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)), y=history.history['loss'],
                                  mode='lines+markers', name='Train Loss'))
    loss_fig.add_trace(
        go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)), y=history.history['val_loss'],
                   mode='lines+markers', name='Validation Loss'))
    loss_fig.update_layout(title=f'{model_name} - Training and Validation Loss', xaxis_title='Epoch',
                           yaxis_title='Loss')
    # Save the loss graph as an image
    loss_image_path = f'../models/glove_trained/{model_name}_loss.png'
    pio.write_image(loss_fig, loss_image_path, format='png')

    # Show the loss graph
    loss_fig.show()

# Plot combined training and validation accuracy using Plotly
accuracy_fig = go.Figure()
for result in results:
    accuracy_fig.add_trace(go.Scatter(x=list(range(1, len(result["history"].history['accuracy']) + 1)),
                                      y=result["history"].history['accuracy'], mode='lines+markers',
                                      name=f'{result["model_name"]} Train Accuracy'))
    accuracy_fig.add_trace(go.Scatter(x=list(range(1, len(result["history"].history['val_accuracy']) + 1)),
                                      y=result["history"].history['val_accuracy'], mode='lines+markers',
                                      name=f'{result["model_name"]} Validation Accuracy'))
accuracy_fig.update_layout(title='Training and Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')

# Save the combined accuracy plot as an image
accuracy_combined_image_path = '../models/glove_trained/combined_accuracy.png'
pio.write_image(accuracy_fig, accuracy_combined_image_path, format='png')

# Plot test accuracy using Plotly
test_accuracy_fig = go.Figure()
for result in results:
    test_accuracy_fig.add_trace(
        go.Bar(x=[result["model_name"]], y=[result["test_accuracy"]], name=f'{result["model_name"]} Test Accuracy'))
test_accuracy_fig.update_layout(title='Test Accuracy for Different Models', xaxis_title='Model',
                                yaxis_title='Accuracy')

# Save the test accuracy plot as an image
test_accuracy_image_path = '../models/glove_trained/test_accuracy.png'
pio.write_image(test_accuracy_fig, test_accuracy_image_path, format='png')

# Show the accuracy plots
accuracy_fig.show()
test_accuracy_fig.show()









