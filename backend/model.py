"""Lightweight ensemble (word embeddings + character CNN) for three-class moderation."""
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(42)

WORD_TOKENS = 40000
WORD_LEN = 120
CHAR_TOKENS = 3000
CHAR_LEN = 200


def build_word_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(WORD_LEN,)),
        tf.keras.layers.Embedding(WORD_TOKENS, 96),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])


def build_char_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(CHAR_LEN,)),
        tf.keras.layers.Embedding(CHAR_TOKENS, 48),
        tf.keras.layers.Conv1D(128, 4, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])


class Ensemble:
    def __init__(self):
        self.word_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=WORD_TOKENS, output_mode="int", output_sequence_length=WORD_LEN)
        self.char_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=CHAR_TOKENS, split="character", output_mode="int", output_sequence_length=CHAR_LEN)
        self.word_model = build_word_model()
        self.char_model = build_char_model()

    def fit(self, texts, labels, epochs=10, batch_size=128):
        texts, labels = np.array(texts), np.array(labels)
        weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels)
        class_weight = {index: weight for index, weight in enumerate(weights)}
        self.word_vectorizer.adapt(texts)
        self.char_vectorizer.adapt(texts)
        for model, vectorizer in ((self.word_model, self.word_vectorizer), (self.char_model, self.char_vectorizer)):
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            model.fit(vectorizer(texts).numpy(), labels, epochs=epochs, batch_size=batch_size,
                      class_weight=class_weight, verbose=0)
        return self

    def predict_proba(self, texts):
        texts = np.array(texts)
        word_probs = self.word_model(self.word_vectorizer(texts), training=False).numpy()
        char_probs = self.char_model(self.char_vectorizer(texts), training=False).numpy()
        return (word_probs + char_probs) / 2
