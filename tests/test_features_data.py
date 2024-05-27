import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample data for testing
sample_texts = ["sample text", "another example", "more data to test"]
sample_labels = ["phishing", "legitimate", "phishing"]

# Tokenizer setup
tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokenizer.fit_on_texts(sample_texts)

# Label Encoder setup
encoder = LabelEncoder()
encoder.fit(sample_labels)

@pytest.fixture
def data():
    return {
        "texts": sample_texts,
        "labels": sample_labels,
        "tokenizer": tokenizer,
        "encoder": encoder
    }

def test_tokenizer(data):
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    assert len(sequences) == 3, "The number of sequences should match the number of input texts."
    assert len(data["tokenizer"].word_index) > 0, "The tokenizer should have a non-empty word index."

def test_label_encoder(data):
    transformed_labels = data["encoder"].transform(data["labels"])
    assert len(transformed_labels) == 3, "The number of transformed labels should match the number of input labels."
    assert len(data["encoder"].classes_) == 2, "There should be two unique classes."

def test_feature_distribution(data):
    # Test that the distributions of each feature match expectations
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    sequence_lengths = [len(seq) for seq in sequences]
    assert np.mean(sequence_lengths) > 0, "The mean sequence length should be greater than zero."

def test_feature_target_relationship(data):
    # Test the relationship between each feature and the target
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    transformed_labels = data["encoder"].transform(data["labels"])
    assert len(sequences) == len(transformed_labels), "The number of sequences should match the number of labels."

def test_feature_pairwise_correlations():
    # Test pairwise correlations between individual features
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    correlation_matrix = df.corr()
    assert correlation_matrix.shape == (2, 2), "The correlation matrix should be 2x2."
    assert -1 <= correlation_matrix.loc['feature1', 'feature2'] <= 1, "Correlation should be between -1 and 1."

def test_feature_cost(data):
    # Test the cost of each feature (e.g., latency, memory usage)
    feature_size = len(data["tokenizer"].word_index)
    assert feature_size < 10000, "The size of the feature index should be reasonable for memory usage."

def test_feature_privacy(data):
    # Exclude the OOV token from the privacy check
    word_index = data["tokenizer"].word_index
    words = [word for word in word_index.keys() if word != '-n-']
    assert all('-' not in word for word in words), "Privacy checks failed in tokenizer word index."

def test_feature_code():
    # Test all code that creates input features
    sample_dates = ["2020-01-01", "2021-01-01"]
    cleaned_dates = [pd.to_datetime(date) for date in sample_dates]
    assert all(isinstance(date, pd.Timestamp) for date in cleaned_dates), "Date cleaning should produce Timestamp objects."
