This project contains 2 separate models to perform Offensive Language Classification.

**1.  Classification with FastText and LSTM**

**2. Classification using Fine-tune BERT**

You can install all the required libraries and dependencies by running:

```python
pip install -r Requirements.txt
```

###1. Offensive Language Classification using FastText and LSTM

This project builds a text classification model to detect toxic comments using FastText embeddings and LSTM. The model is trained to identify toxicity in user comments and can be evaluated on both labeled and unlabeled test data.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Keras
- scikit-learn
- Pandas
- Matplotlib, Seaborn
- FastText
- tqdm
- WordCloud

#### Dataset
**train.csv:** The training dataset contains feedback text and toxicity labels (toxic, abusive, etc.).

**validation.csv:** A validation dataset for tuning model performance.

**test.csv:** The test dataset to evaluate the final model's performance.

**test_labels.csv:** Labels corresponding to the test dataset (optional for evaluation).


#### Features
- Data Preprocessing:
![Visualization of Train Data](https://github.com/PromitSarker/Offensive-Language-Classification/blob/main/toxic_non_toxic.png "Visualization of Train Data")

- Tokenization and padding of text.

- Stopwords removal.

- Feature engineering: word count, unique word count, mean word length, etc.

#### Model:

- Embedding layer using FastText pre-trained word embeddings.

- Bidirectional LSTM layer for sequence modeling.

- Dense layers with dropout for regularization.

#### Training:

- Early stopping to avoid overfitting.

- Training with TPU support if available.

#### Evaluation:

- Classification report and accuracy on test data.

- Confusion matrix visualization.

- Language-wise performance analysis
![Langugae wise performance](https://github.com/PromitSarker/Offensive-Language-Classification/blob/main/language_performance.png "Langugae wise performance")

#### How to Run

```python
git clone https://github.com/PromitSarker/Offensive-Language-Classification.git

```
```python
cd Offensive-Language-Classification
```
Or,

- **Prepare the dataset:** Ensure the datasets are placed in the Dataset/ folder.

- **Run the script:** Execute the main script to train the model:

- **Evaluate performance:** After training, the model will evaluate the test data. If labels are present, it will display the accuracy, classification report, and confusion matrix.

- **Predictions:** If no labels are provided for test data, the model will save predictions in a test_predictions.csv file.

#### Results
- The model uses FastText embeddings for multi-language support.

- Supports TPU training for faster execution (if available).

- Provides detailed performance analysis on a per-language basis.

.
.
.
.

###2. Offensive Language Classification using Fine-tune BERT

This code is for classifying toxic comments in multilingual text using the BERT (Bidirectional Encoder Representations from Transformers) model. The project uses TensorFlow and Hugging Face's transformers library to fine-tune a pre-trained BERT model on a dataset of toxic comments.

#### Data Preparation
**Loading Datasets:** The datasets are loaded from CSV files:

- train.csv - Training data

- validation.csv - Validation data

- test.csv - Test data

- test_labels.csv - Labels for the test data

**Data Preprocessing:** Toxicity labels are computed from multiple columns, and text is tokenized using a pre-trained BERT tokenizer.

**Model**
The model uses BERT for feature extraction, with a custom classification head for binary toxicity prediction.

```python
def build_model(transformer_model, max_len=96):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    cls_output = Lambda(extract_cls_token, output_shape=(transformer_model.config.hidden_size,))([input_ids, attention_mask])
    output = Dense(1, activation='sigmoid')(cls_output)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    return model
```
#### **Training**
The model is trained using the train_dataset and validation_dataset. Early stopping and model checkpointing are used during training to avoid overfitting and save the best model.

#### Evaluation
#### Predictions
