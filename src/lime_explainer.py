"""
LIME wrappers for the multi-task CNN.

LIME normally expects a classifier, but our model outputs continuous scores.
The trick: duplicate each score into a 2-column array so LIME's internals
don't complain, then tell it to explain "label 1".
"""

import numpy as np
from lime.lime_text import LimeTextExplainer

from src.preprocessing import clean_for_nn, MAX_LEN


def predict_difficulty_for_lime(texts, model, tokenizer):
    """
    Prediction wrapper for LIME -- Difficulty version.

    Takes raw text, runs it through the model, and returns a 2D array
    where column 1 is the predicted difficulty score.

    Args:
        texts: List of raw review strings.
        model: Trained multi-task CNN.
        tokenizer: Fitted Keras Tokenizer.

    Returns:
        Array of shape (n_samples, 2). Column 0 is a dummy.
    """
    cleaned_texts = [clean_for_nn(text) for text in texts]
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    predictions = model.predict(padded, verbose=0)

    # Index 1 is the Difficulty head
    difficulty_preds = predictions[1].reshape(-1)

    return np.vstack((difficulty_preds, difficulty_preds)).T


def predict_quality_for_lime(texts, model, tokenizer):
    """
    Prediction wrapper for LIME -- Quality version.

    Same as predict_difficulty_for_lime but grabs the Quality head (index 0).

    Args:
        texts: List of raw review strings.
        model: Trained multi-task CNN.
        tokenizer: Fitted Keras Tokenizer.

    Returns:
        Array of shape (n_samples, 2). Column 1 is the quality score.
    """
    cleaned_texts = [clean_for_nn(text) for text in texts]
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    predictions = model.predict(padded, verbose=0)

    # Index 0 is the Quality head
    quality_preds = predictions[0].reshape(-1)

    return np.vstack((quality_preds, quality_preds)).T


def explain_difficulty(text, model, tokenizer, num_features=6):
    """
    Explain what words made the model predict a certain difficulty score.

    Args:
        text: Raw review string.
        model: Trained multi-task CNN.
        tokenizer: Fitted Keras Tokenizer.
        num_features: How many words to show (default 6).

    Returns:
        LIME Explanation object -- use exp.as_list(label=1) to get
        (word, weight) pairs.
    """
    explainer = LimeTextExplainer(
        class_names=['Ignored', 'Difficulty_Score']
    )

    exp = explainer.explain_instance(
        text,
        lambda texts: predict_difficulty_for_lime(texts, model, tokenizer),
        labels=(1,),
        num_features=num_features
    )
    return exp


def explain_quality(text, model, tokenizer, num_features=6):
    """
    Explain what words made the model predict a certain quality score.

    Args:
        text: Raw review string.
        model: Trained multi-task CNN.
        tokenizer: Fitted Keras Tokenizer.
        num_features: How many words to show (default 6).

    Returns:
        LIME Explanation object -- use exp.as_list(label=1) to get
        (word, weight) pairs.
    """
    explainer = LimeTextExplainer(
        class_names=['Ignored', 'Quality_Score']
    )

    exp = explainer.explain_instance(
        text,
        lambda texts: predict_quality_for_lime(texts, model, tokenizer),
        labels=(1,),
        num_features=num_features
    )
    return exp
