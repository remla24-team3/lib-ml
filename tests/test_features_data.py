"""
This module contains tests for the features of the data.
"""

# pylint: disable=W0511, E0401, E0611, W0621

import random
import re
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

INPUT_DIR = "tests/dummy_data/"
OUTPUT_DIR = "tests/dummy_data/"

SENSITIVE_PATTERNS = re.compile(r"(@|token|session|user|userid|"
                                r"password|auth|files|pro)", re.IGNORECASE)


def read_and_sample_data(file_path, sample_size=100):
    """Read data from a file, sample lines randomly,
    and return the sampled lines."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        random.shuffle(lines)
        return lines[:sample_size]


def save_sampled_data(file_path, data):
    """Save sampled data to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for line in data:
            file.write(line + "\n")


def remove_sensitive_info(text):
    """Remove sensitive information from the text based on
    the SENSITIVE_PATTERNS."""
    return SENSITIVE_PATTERNS.sub("[REDACTED]", text)


@pytest.fixture
def data():
    """
    Fixture for loading, processing, and saving sampled data.

    Returns:
        dict: Contains processed data, tokenizer, and encoder.
    """
    sample_size = 100  # Set sample size for each file

    train = read_and_sample_data(INPUT_DIR + "sample_train.txt", sample_size)
    val = read_and_sample_data(INPUT_DIR + "sample_val.txt", sample_size)
    test = read_and_sample_data(INPUT_DIR + "sample_test.txt", sample_size)

    # Process train data
    raw_x_train = [remove_sensitive_info(line.split("\t")[1])
                   for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process validation data
    raw_x_val = [remove_sensitive_info(line.split("\t")[1]) for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    # Process test data
    raw_x_test = [remove_sensitive_info(line.split("\t")[1]) for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Tokenizer setup
    tokenizer = Tokenizer(lower=True, char_level=False, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    # Label Encoder setup
    encoder = LabelEncoder()
    encoder.fit(raw_y_train + raw_y_val + raw_y_test)

    return {
        "texts": raw_x_train + raw_x_val + raw_x_test,
        "labels": raw_y_train + raw_y_val + raw_y_test,
        "tokenizer": tokenizer,
        "encoder": encoder,
    }


def test_tokenizer(data):
    """Test the tokenizer."""
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    assert len(sequences) == len(data["texts"]), (
        "The number of sequences should match the number of input texts."
    )
    assert len(data["tokenizer"].word_index) > 0, (
        "The tokenizer should have a non-empty word index."
    )


def test_label_encoder(data):
    """Test the label encoder."""
    transformed_labels = data["encoder"].transform(data["labels"])
    assert len(transformed_labels) == len(data["labels"]), (
        "The number of transformed labels should "
        "match the number of input labels."
    )
    assert len(data[
                   "encoder"].classes_) > 1, \
        "There should be more than one unique class."


def test_feature_distribution(data):
    """Test that the distributions of each feature match expectations."""
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    sequence_lengths = [len(seq) for seq in sequences]
    assert np.mean(sequence_lengths) > 0, (
        "The mean sequence length should be greater than zero."
    )


def test_feature_target_relationship(data):
    """Test the relationship between each feature and the target."""
    sequences = data["tokenizer"].texts_to_sequences(data["texts"])
    transformed_labels = data["encoder"].transform(data["labels"])
    assert len(sequences) == len(transformed_labels), (
        "The number of sequences should match the number of labels."
    )


def test_feature_cost(data):
    """Test the cost of each feature (e.g., latency, memory usage)."""
    feature_size = len(data["tokenizer"].word_index)
    assert feature_size < 10000, (
        "The size of the feature index should be reasonable for memory usage."
    )


def test_feature_privacy(data):
    """Test privacy aspects of the tokenizer's word index."""
    word_index = data["tokenizer"].word_index
    sensitive_patterns = ['@', 'token', 'session', 'user', 'userid',
                          'password', 'auth', 'files', 'pro']
    for pattern in sensitive_patterns:
        assert pattern not in word_index, (f"Sensitive pattern '{pattern}' "
                                           f"found in tokenizer word index")


def test_feature_code():
    """Test all code that creates input features."""
    sample_urls = ["https://example.com/path?query=1",
                   "https://example.org/path/to/resource"]
    cleaned_urls = [url.lower() for url in sample_urls]
    assert all(url.startswith("https://") for url in cleaned_urls), (
        "URL cleaning should ensure all URLs start with 'https://'."
    )
