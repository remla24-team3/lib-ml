# lib-ml

`lib-ml` is a Python library designed for preprocessing text data, especially tailored for machine learning applications. The library offers robust tools for tokenization, sequence padding, and label encoding, ensuring that text data is optimally prepared for model training and analysis. The library is available on PyPI and can be easily integrated into your projects. New releases are automatically created and pushed to PyPI with every push to main.

##  Repository Structure

```sh
└── lib-ml/
    ├── .github
    │   └── workflows
    ├── LICENSE
    ├── README.md
    ├── dist
    │   ├── lib_ml_group3-0.2.0-py3-none-any.whl
    │   └── lib_ml_group3-0.2.0.tar.gz
    ├── lib_ml_group3
    │   ├── __init__.py
    │   ├── load_model.py
    │   └── preprocessing.py
    ├── poetry.lock
    └── pyproject.toml
```

## Installation

Install `lib-ml` from PyPI:

```bash
pip install lib-ml-group3
```

## Features

`lib-ml` includes the following features:
- **Data Tokenization**: Convert text into sequences of tokens or characters.
- **Sequence Padding**: Pad sequences to a uniform length to ensure consistency among data inputs.
- **Label Encoding**: Encode labels in a way that is suitable for machine learning models.
- **Persistence**: Save and load preprocessed data using Python's pickle module for easy reusability.

## Usage

Here is a quick example of how to use `lib-ml` for text data preprocessing:

```python
from lib_ml.preprocessing import preprocess_data

# Preprocess the data and save it to disk
preprocess_data()
```

The `preprocess_data()` function reads data from specified input directories, processes the text and labels, and saves the tokenized and encoded outputs to designated output directories.

## License

`lib-ml` is open source software [licensed as MIT](LICENSE).

## Support

If you have any questions or issues with `lib-ml`, please open an issue on the project repository, and we will get back to you as soon as possible.