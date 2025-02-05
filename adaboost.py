import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralNetworkEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=5, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.input_dim, activation='relu'),
            tf.keras.layers.Dense(64, input_dim=self.input_dim, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.model.fit(X, y, sample_weight=sample_weight, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)


def load_data(data_dir="./src/data/", train_file="./src/train.txt"):
    train_df = pd.read_csv(train_file, sep=",", header=0, names=["guid", "tag"])

    texts = []
    labels = []

    for idx, row in train_df.iterrows():
        guid = str(row["guid"])
        tag = row["tag"]

        txt_path = os.path.join(data_dir, guid + ".txt")

        if not os.path.isfile(txt_path):
            print(f"找不到文件 {txt_path}")
            continue

        with open(txt_path, "r", encoding="ascii", errors="replace") as f:
            text_content = f.read().strip()

        texts.append(text_content)
        labels.append(tag)

    return texts, labels


def preprocess_texts_and_labels(texts, labels, vocab_size=5000, max_len=50):
    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    numeric_labels = [label_mapping[label] for label in labels]
    y_data = np.array(numeric_labels)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x_data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return x_data, y_data, tokenizer


def main():
    data_dir = "./src/data/"
    train_file = "./src/train.txt"
    texts, labels = load_data(data_dir, train_file)
    print(f"共读取到 {len(texts)} 条文本数据。")

    x_data, y_data, tokenizer = preprocess_texts_and_labels(texts, labels, vocab_size=50000, max_len=120)
    print("文本序列形状:", x_data.shape)
    print("标签形状:", y_data.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    print("训练集大小:", x_train.shape[0])
    print("验证集大小:", x_val.shape[0])

    nn_estimator = NeuralNetworkEstimator(input_dim=x_train.shape[1], epochs=10, batch_size=32)

    ada_boost = AdaBoostClassifier(base_estimator=nn_estimator, n_estimators=20)

    ada_boost.fit(x_train, y_train)

    val_acc = ada_boost.score(x_val, y_val)
    print(f"验证集准确率: {val_acc:.4f}")


if __name__ == "__main__":
    main()
