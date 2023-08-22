## Sentiment Analysis With Naive Bayes
![image title](https://img.shields.io/badge/Python-v3.x-green.svg) ![image title](https://img.shields.io/badge/Streamlit-v1.23.0-red.svg) ![Image title](https://img.shields.io/badge/NLTK-v3.6.7-orange.svg) ![image title](https://img.shields.io/badge/Pandas-v2.0-blue.svg)![image title](https://img.shields.io/badge/tensorflow-v2.13.0-orange.svg)

<br>
<br>
<p align="center">
  <img src="./images/sentimentanalysishotelgeneric-2048x803-1.jpg" 
       width="1200">
</p>
<br>


## Table of Contents
1. [Introduction](#introduction)
   
   1.1 [Project Description](#discription)

   1.2 [Project Motivation](#motivation)

2. [Project Overview](#project_overview)

   2.1 [Overview Dataset](#datasetoverview)

   2.2 [Dataset Problem statement](#problemstatement)

3. [Features](#Features)

4. [Project Directory Structure](#DirectoryStructure)

5. [Steps](#Steps)

   5.1 [Data Collection and Preprocessing](#exploratory-data-analysis)
   
   5.2 [Calculating Word Counts and Likelihoods](#data-preprocessing)

   5.3 [Train-Test Split and Model Training](#model-development)

   5.4 [Running the App](#evaluation-and-metrics)

   5.5 [Train-Test Split and Model Training](#model-development)

   5.6 [Interact with the App](#evaluation-and-metrics)

6. [Requirements](#Requirements)
7. [Usage](#usage)
8. [Screenshots](#Screenshots)
9. [EDA Notebook Structure](#EDA)
   
   9.1 [Running the Notebook](#exploratory-data-analysis1)

   9.2 [Results and Visualizations](#exploratory-data-analysis2)
10. [License](#License)
11. [Acknowledgements](#Acknowledgements)
12. [Contact Information](#Contact)

## Introduction <a name="introduction"></a>

### Project Description <a name="discription"></a>

Sentiment analysis is the process of determining the sentiment (positive, negative, neutral) expressed in a piece of text. This project utilizes various deep learning models and pre-trained word embeddings (GloVe) to perform sentiment analysis on textual data. The primary goal is to classify text into one of three categories: positive, negative, or neutral sentiment.

### Project Motivation <a name="motivation"></a>
Sentiment analysis is a fundamental task in natural language processing that has various real-world applications. Understanding the sentiment expressed in text data can provide valuable insights into user opinions, emotions, and trends. This project was motivated by the desire to explore sentiment analysis techniques and showcase their implementation through an interactive web application.

The goals of this project include:

- `Exploring Deep Learning:` Our primary objective is to delve into the domain of deep learning, with a specific focus on recurrent neural networks (RNNs). These RNNs demonstrate remarkable proficiency in comprehending textual data. Additionally, we are actively engaged in experimenting with diverse approaches to determine the most effective one.
- `Enhancing Text Understanding:`  In our endeavor to enhance the intelligence of our models, we are incorporating advanced word techniques. Notably, we are leveraging GloVe, a technique that significantly amplifies our models' capacity to comprehend the semantic nuances of words.
- `User-Friendly App:` We're packaging all these advanced models into a user-friendly application using Streamlit. This means anyone can use it to get insights from their text.

By sharing this project, we aim to contribute to the knowledge and understanding of sentiment analysis while providing a hands-on example for those interested in exploring natural language processing and interactive web application development.

## Project Overview <a name="project_overview"></a>
### Overview of the Dataset <a name="datasetoverview"></a>

The dataset used for this project is the "Tweet Sentiment Extraction" dataset from Kaggle. This dataset contains tweets along with their associated sentiment labels and selected text. The selected text provides a concise representation of the tweet's sentiment. The dataset is utilized to train sentiment analysis models for predicting the sentiment of tweets.

#### Columns

- `textID`: A unique ID for each piece of text.
- `text`: The text of the tweet.
- `sentiment`: The general sentiment label of the tweet (positive, negative, or neutral).
- `selected_text` (Training only): The text that supports the tweet's sentiment, serving as a sentiment indicator.

### Dataset Problem statement <a name="problemstatement"></a>

Given the text of a tweet, the task is to classify the sentiment as `positive`, `negative`, or `neutral`. This involves training a model to understand the emotional tone of the text.




# Project Directory Structure <a name="Structure"></a>
```bash
│                      
├── app.py                    # Streamlit application script
├── data                      # Directory for storing the dataset
│   ├── glove
│   │   └── glove.6B.100d.txt
│   │   └── max_seq_length.txt
│   │   └── tokenizer_word_index.npy
│   └── train
│       └── train.csv
│
├── images                     # Directory for sentiment image
│   ├── app_Sentiment_1.jpg    # web app screenshot 1
│   └── app_Sentiment_2.jpg    # web app screenshot 2
│   └── app_Sentiment_3.jpg    # web app screenshort 3
│   └── negative.jpg           # Positive sentiment image
│   └── neutral.jpg            # Positive sentiment image
│   └── positive.jpg           # Positive sentiment image
│   └── sentimentanalysishotelgeneric-2048x803-1.jpg
├── models
│   └── glove_trained
├── .gitignore                       # ignore files that cannot commit to Git
├── notebooks                        # store notebooks
│   └── EDA_sentiment_analysis.ipynb # EDA Notebook
├── logs.txt                         # Streamlit log files 
├── requirements.txt                 # List of required packages
├── README.md                        # Project README file
├── LICENSE                          # Project license
```

## Description <a name="Description"></a>

This project is centered around sentiment analysis, which involves discerning the sentiment (positive, negative, or neutral) conveyed in a text. To achieve this, we employ a comprehensive range of RNN architectures to classify the sentiment of input text and present the outcomes through an easily accessible Streamlit application.

## Features <a name="Features"></a>

- Text Data Preprocessing: The system preprocesses textual data by tokenizing and removing links, enhancing data quality.
- Versatile RNN Models: It utilizes TensorFlow to create a diverse set of RNN models, expanding the range of available architectures.
- Model Visualization: The application displays detailed model architecture and summaries, aiding in model understanding.
- Sentiment Classification: It classifies sentiment in the input text and provides sentiment scores, enhancing text summary.
- Visual Representation: The system generates resized sentiment-specific images based on predicted sentiment, enriching user experience.
- User-Friendly Interface: The application offers an aesthetically pleasing and intuitive layout for seamless user interaction.

## Steps <a name="Steps"></a>

1. **Data Collection:** 
   - Gather a dataset from [Kaggle](https://www.kaggle.com/competitions/tweet-sentiment-extraction) containing positive, negative, and neutral sentiment-labeled text.
   - We need to download Stanford's GloVe 100-dimensional (100d) word embeddings from the [Kaggle](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt) and place it in `data/glove/glove.6B.100d.txt`.
2. **Data Splitting:**
   - The dataset `(df)` is split into three parts: `training`, `validation`, and test sets using `train_test_split`.
   - The training set `(train_df)` contains 80% of the data, while the validation set `(val_df)` and test set `(test_df)` each contain 10% of the data.

3. **Model Training:**
     Multiple model architectures are trained on the sentiment analysis task, providing a diverse set of approaches to the problem. These models include:
   - Bidirectional GRU (`model_gru`)
   - Bidirectional LSTM (`model_lstm`)
   - Stacked Bidirectional LSTM (`model_lstm2`)
   - 1D Convolutional Neural Network (`model_conv`)
   - Flattened Embedding with Dense Layers (`model_flatten`)
   - 1D Convolutional Neural Network with Global Max Pooling (`model_lstm_con`)
   - 1D Convolutional Neural Network with Global Max Pooling and Attention (`model_with_attention`).
     Each of these models offers a unique architecture for sentiment analysis, contributing to a comprehensive evaluation of their performance.

4. **Training and Evaluation:**
   - All models are trained using the Adam optimizer and the sparse categorical cross-entropy loss function.
   - The training process includes early stopping with a patience of 10 epochs to monitor validation accuracy and restore the best weights.
   - ModelCheckpoint is used to save the best-performing model during training based on validation accuracy.

5. **Displaying Model Summary and Architecture:**
   - For each model, both the model summary and architecture diagram are displayed.
   - The model architecture diagrams are saved as images in the `../models/glove_trained/` directory.

6. **Plotting Training and Validation Metrics:**
   - Individual accuracy and loss plots for training and validation data are created using Plotly and saved as images.
   - A combined accuracy plot for all models and a test accuracy bar plot are also created and saved as images.

7. **Saving Best Model:**
   - The best-performing model during training (based on validation accuracy) is saved as an HDF5 file with a custom name, such as `best_model_gru.h5`.
   - These best models are saved in the `../models/glove_trained/` directory for later use.

8. **Saving Best Model:**
   - All models are trained using the Adam optimizer and the sparse categorical cross-entropy loss function.
   - The training process includes early stopping with a patience of 10 epochs to monitor validation accuracy and restore the best weights.
   - ModelCheckpoint is used to save the best-performing model during training based on validation accuracy.

9. **Create Streamlit App Layout:**
   - Build a Streamlit web application for user interaction.
   - Incorporate text input, sentiment classification, and display of sentiment scores.
   - Display sentiment-specific images based on the predicted sentiment.
   
10. **Running the App:**
    - Install the required packages using `pip install streamlit pandas nltk`.
    - Run the Streamlit app using `streamlit run app.py`.

11. **Interact with the App:**
    - Enter text in the provided text area.
    - Select a sentiment analysis model from the dropdown. 
    - Click the "Classify Sentiment" button.
    - View the predicted sentiment label, model architecture, model summary, class probabilities, and a corresponding sentiment image.

## Requirements <a name="Requirements"></a>

- Python 3.x
- Streamlit
- Pandas
- NLTK (Natural Language Toolkit)
- Plotly
- Scikit-learn (sklearn)
- TensorFlow
- Python
- Graphviz
- Pydot
- Kaleido

## Usage <a name="Usage"></a>

1. Clone this repository:
```bash
git clone https://github.com/zubairashfaque/Sentiment-Analysis-with-Naive-Bayes-Streamlit.git
```
2. Run the following command to create a virtual environment named "Sentiment_testing" (you can replace "Sentiment_testing" with any name you prefer):
```bash
python -m venv Sentiment_testing
```
3. To activate the virtual environment, use the following command:
```bash
Sentiment_testing\Scripts\activate
```
4. Install the required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
5. Run the Streamlit app:
```bash
streamlit run app.py
```

6. Enter text in the provided text area and click the "Classify Sentiment" button to see the sentiment prediction and scores.

## Screenshots <a name="Screenshots"></a>

<br>
<br>
<p align="center">
  <img src="./images/app_Sentiment_1.1.jpg" 
       width="1200">
</p>
<br>

<br>
<br>
<p align="center">
  <img src="./images/all models.jpg" 
       width="1200">
</p>
<br>
<br>
<br>
<p align="center">
  <img src="./images/1_prediction.jpg" 
       width="1200">
</p>
<br>


## Notebook Structure <a name="EDA"></a>
The Jupyter Notebook (`EDA_sentiment_analysis.ipynb`) is structured as follows:

1. **Introduction and Setup:** Importing libraries and loading the dataset.
2. **Data Exploration:** Displaying basic dataset information.
3. **Sentiment Distribution Visualization:** Visualizing the distribution of sentiment labels.
4. **Text Preprocessing:** Defining preprocessing functions for tokenization and stemming.
5. **Word Count Analysis:** Calculating word counts for different sentiment classes.
6. **Top Words Visualization:** Displaying top words for each sentiment class and creating treemap visualizations.

## Running the Notebook <a name="exploratory-data-analysis1"></a>
Follow these steps to run the `EDA_sentiment_analysis.ipynb` notebook:

1. Ensure you have Python and the required libraries installed.
2. Open the notebook using Jupyter Notebook or Jupyter Lab.
3. Execute each cell sequentially to see the analysis results.

## Results and Visualizations <a name="exploratory-data-analysis2"></a>
The notebook produces various insightful visualizations, including:
- Sentiment distribution using a Funnel-Chart.
- Top words and their counts for positive, negative, and neutral sentiments.
- Treemap visualizations of top words for each sentiment class.

Sample images of these visualizations are provided in the repository's `images` folder.

## License <a name="License"></a>

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements <a name="Acknowledgements"></a>

- The sentiment analysis algorithm is based on the Naïve Bayes approach.
- Streamlit is used for creating the user interface.
- NLTK is used for text preprocessing.

## Contact Information <a name="Contact"></a>
For questions, feedback, or discussions related to this project, you can contact me at [mianashfaque@gmail.com](mailto:mianashfaque@gmail.com).
