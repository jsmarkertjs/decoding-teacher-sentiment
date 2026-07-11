"""
Turns raw RateMyProfessor reviews into padded sequences the model can eat.

Handles cleaning, tokenization, padding, and train/test splitting.
"""

import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# --- Constants ---
MAX_WORDS = 10000   # Keep the 10,000 most common words
MAX_LEN = 150       # Pad or truncate all reviews to 150 tokens


def clean_for_nn(text):
    """
    Clean up review text before feeding it to the network.

    Lowercases everything, strips punctuation/numbers/special chars,
    and collapses extra whitespace.

    Args:
        text: Raw review as a string.

    Returns:
        Cleaned string with only lowercase letters and spaces.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_tokenizer(texts):
    """
    Create and fit a Keras Tokenizer on a list of cleaned texts.

    Args:
        texts: Iterable of raw review strings (they get cleaned internally).

    Returns:
        A fitted Tokenizer (num_words=MAX_WORDS, OOV token="<OOV>").
    """
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def prepare_features(texts, tokenizer, maxlen=MAX_LEN):
    """
    Convert cleaned text into padded integer sequences.

    Args:
        texts: Iterable of already-cleaned text strings.
        tokenizer: A fitted Keras Tokenizer.
        maxlen: Max sequence length (default 150).

    Returns:
        Numpy array of shape (n_samples, maxlen).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, padding='post')


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Run the full preprocessing pipeline from raw DataFrame to train/test splits.

    What it does:
        1. Drops rows missing review_text
        2. Combines review_text + tags into full_text
        3. Cleans everything with clean_for_nn()
        4. Fits a tokenizer and converts to padded sequences
        5. Splits 80/20 into train and test sets

    Args:
        df: DataFrame with columns 'review_text', 'tags', 'quality', 'difficulty'.
        test_size: Fraction to hold out for testing (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - X_train, X_test: padded sequences
            - y_q_train, y_q_test: quality targets
            - y_d_train, y_d_test: difficulty targets
            - tokenizer: the fitted Tokenizer
            - vocab_size: how many unique words we found
    """
    df = df.dropna(subset=['review_text']).copy()

    # Stick tags onto the end of the review text -- they're surprisingly useful
    df['tags'] = df['tags'].fillna('')
    df['full_text'] = df['review_text'] + " " + df['tags']

    df['nn_cleaned_text'] = df['full_text'].apply(clean_for_nn)

    tokenizer = build_tokenizer(df['nn_cleaned_text'])
    X = prepare_features(df['nn_cleaned_text'], tokenizer)

    y_quality = df['quality'].values
    y_difficulty = df['difficulty'].values

    X_train, X_test, y_q_train, y_q_test, y_d_train, y_d_test = train_test_split(
        X, y_quality, y_difficulty,
        test_size=test_size,
        random_state=random_state
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_q_train': y_q_train,
        'y_q_test': y_q_test,
        'y_d_train': y_d_train,
        'y_d_test': y_d_test,
        'tokenizer': tokenizer,
        'vocab_size': len(tokenizer.word_index),
    }
