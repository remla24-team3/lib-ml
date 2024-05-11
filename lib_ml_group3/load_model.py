# pylint: disable=E0611,E0401,R0903

"""
Tokenization utilities for text data preprocessing.
"""

import pickle
import keras

from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_model(model_path, encoder_path, tokenizer_path):
    """ Load the model from the given paths.

    Args:
        model_path (str): Path to the model file.
        encoder_path (str): Path to the encoder file.
        tokenizer_path (str): Path to the tokenizer file.

    Returns:
        Model: Model object loaded from the given paths.
    """
    return Model(model_path, encoder_path, tokenizer_path)


class Model:
    """ Model wrapper class """

    def __init__(self, model_path, encoder_path, tokenizer_path):
        self.model = keras.models.load_model(model_path)
        with open(encoder_path, "rb") as encoder_file:
            self.encoder = pickle.load(encoder_file)
        with open(tokenizer_path, "rb") as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

    def predict(self, url_list: list[str]):
        """ Predict the class of the given URLs.

        Args:
            url_list (list[str]): List of URLs to predict.

        Returns:
            ndarray[str]: Predicted classes of the URLs.
        """
        sequence_length = 200
        tokenized = pad_sequences(
            self.tokenizer.texts_to_sequences(url_list),
            maxlen=sequence_length
        )
        prediction = self.model.predict(tokenized)
        return self.encoder.inverse_transform(prediction.round().astype(int))
