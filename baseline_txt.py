import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_data(data_dir="./src/data/", train_file="./src/train.txt"):
    """
    读取 train.txt 文件（guid, tag），并根据每个 guid，
    在 data_dir 下找到对应的 guid.txt 文件，读取其文本。

    返回:
    texts: list[str], 每条 guid 对应的文本内容
    labels: list[str], 每条 guid 对应的标签(negative/neutral/positive)
    """
    train_df = pd.read_csv(train_file,
                           sep=",",
                           header=0,
                           names=["guid", "tag"])

    texts = []
    labels = []

    for idx, row in train_df.iterrows():
        guid = str(row["guid"])
        tag = row["tag"]


        txt_path = os.path.join(data_dir, guid + ".txt")


        if not os.path.isfile(txt_path):
            print(f"警告: 找不到文件 {txt_path}，跳过该条数据。")
            continue

        with open(txt_path, "r", encoding="ascii",errors="replace") as f:
            text_content = f.read().strip()

        texts.append(text_content)
        labels.append(tag)

    return texts, labels


def preprocess_texts_and_labels(texts, labels, vocab_size=5000, max_len=50):
    """
    将文本进行分词、转换为序列并补齐；将标签转换为数字编码(0,1,2)。
    返回: (x_data, y_data, tokenizer)
    """

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    numeric_labels = [label_mapping[label] for label in labels]
    y_data = np.array(numeric_labels)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    x_data = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return x_data, y_data, tokenizer


def build_model(vocab_size=5000, embedding_dim=64, max_len=50):
    """
    构建一个简单的三分类神经网络:
    Embedding -> GlobalAveragePooling1D -> Dense -> Dense(3)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,
                                  output_dim=embedding_dim,
                                  input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 三分类输出
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    data_dir = "./src/data/"
    train_file = "./src/train.txt"

    texts, labels = load_data(data_dir, train_file)
    print(f"共读取到 {len(texts)} 条文本数据。")

    x_data, y_data, tokenizer = preprocess_texts_and_labels(texts, labels,
                                                            vocab_size=5000,
                                                            max_len=50)
    print("文本序列形状:", x_data.shape)
    print("标签形状:", y_data.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.1, random_state=42
    )
    print("训练集大小:", x_train.shape[0])
    print("验证集大小:", x_val.shape[0])
    model = build_model(vocab_size=5000, embedding_dim=64, max_len=x_data.shape[1])
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=32
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"验证集准确率: {val_acc:.4f}")


if __name__ == "__main__":
    main()
