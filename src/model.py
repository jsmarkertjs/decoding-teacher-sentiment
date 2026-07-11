"""
Multi-task 1D CNN that predicts Quality and Difficulty from the same review.

Architecture:
    Embedding(10000 -> 100d)
    -> Conv1D(128 filters, kernel_size=5, relu)
    -> GlobalMaxPooling1D
    -> Dropout(0.5)
    -> [Dense(64, relu) -> Dense(1, linear)]  -- Quality head
    -> [Dense(64, relu) -> Dense(1, linear)]  -- Difficulty head

About 1,080,770 parameters total.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)


def build_model(max_words=10000, max_len=150):
    """
    Build and compile the multi-task CNN.

    Args:
        max_words: How many words in the vocabulary (for the Embedding layer).
        max_len: Length of input sequences (150 tokens).

    Returns:
        A compiled Keras Model with two outputs:
            - Quality_Output (linear, 1 neuron)
            - Difficulty_Output (linear, 1 neuron)
    """
    # Shared layers -- these learn general patterns in the text
    input_layer = Input(shape=(max_len,), name='Text_Input')
    shared_embedding = Embedding(
        input_dim=max_words,
        output_dim=100,
        input_length=max_len
    )(input_layer)
    shared_conv = Conv1D(
        filters=128, kernel_size=5, activation='relu'
    )(shared_embedding)
    shared_pool = GlobalMaxPooling1D()(shared_conv)
    shared_dropout = Dropout(0.5)(shared_pool)

    # Quality head -- learns what makes a review say "good teacher"
    q_dense = Dense(64, activation='relu')(shared_dropout)
    quality_output = Dense(1, activation='linear', name='Quality_Output')(q_dense)

    # Difficulty head -- learns what makes a review say "hard class"
    d_dense = Dense(64, activation='relu')(shared_dropout)
    difficulty_output = Dense(
        1, activation='linear', name='Difficulty_Output'
    )(d_dense)

    model = Model(
        inputs=input_layer,
        outputs=[quality_output, difficulty_output]
    )

    model.compile(
        optimizer='adam',
        loss={'Quality_Output': 'mse', 'Difficulty_Output': 'mse'},
        metrics={'Quality_Output': 'mae', 'Difficulty_Output': 'mae'}
    )

    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
