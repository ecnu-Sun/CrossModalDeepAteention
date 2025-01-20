import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

##############################################################################
# 1. 读取数据：假设 train.txt 中有两列 guid, tag(= negative/neutral/positive)
##############################################################################

def load_text_and_image(
    txt_dir="./src/data/",
    img_dir="./src/data/",
    train_file="./src/train.txt",
    img_height=224,
    img_width=224
):
    """
    读取 train.txt（列: guid, tag），对每个 guid:
      - 文本: guid.txt
      - 图像: guid.jpg
    返回: texts(list[str]), images(numpy.array), labels(numpy.array)
    """
    df = pd.read_csv(train_file, sep=",", header=0, names=["guid","tag"])
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["tag"].map(label_map)

    texts = []
    images = []
    labels = []

    for idx, row in df.iterrows():
        guid = str(row["guid"])
        label = row["label"]

        txt_path = os.path.join(txt_dir, guid + ".txt")
        img_path = os.path.join(img_dir, guid + ".jpg")

        if not os.path.isfile(txt_path):
            print(f"文本文件不存在: {txt_path}")
            continue
        if not os.path.isfile(img_path):
            print(f"图片文件不存在: {img_path}")
            continue

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text_content = f.read().strip()
        texts.append(text_content)

        img = load_img(img_path, target_size=(img_height, img_width))
        img_arr = img_to_array(img)  # shape=(224,224,3)
        images.append(img_arr)

        labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    return texts, images, labels

##############################################################################
# 2. 构建 预训练BERT 的文本分类模型
##############################################################################

def build_pretrained_text_model(model_name="bert-base-uncased", num_classes=3):
    """
    1) 使用 Hugging Face Transformers 的 BERT 模型做句向量表示
    2) 在其输出上接一个简单的分类层
    3) 默认为冻结 BERT，只训练顶层，若想深度微调可调节 `bert_model.trainable=True`
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = TFBertModel.from_pretrained(model_name)

    bert_model.trainable = False

    # 构建 Keras 模型: 输入: dict{"input_ids","attention_mask"} -> BERT -> Dense
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # BERT 输出: last_hidden_state & pooler_output
    outputs = bert_model([input_ids, attention_mask])
    pooled_output = outputs.pooler_output   # shape=(batch,768) for base model

    x = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
    logits = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, tokenizer

##############################################################################
# 3. 构建 预训练ResNet 的图像分类模型
##############################################################################

def build_pretrained_image_model(num_classes=3, img_height=224, img_width=224):
    """
    使用预训练的 ResNet50 (ImageNet 权重) 做特征提取，顶层接一个全连接层进行三分类。
    默认为冻结 ResNet 主干，仅训练顶层，若想微调可设置 base_model.trainable=True
    """
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.layers import Dense, Input

    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False


    inputs = Input(shape=(img_height, img_width, 3))

    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


##############################################################################
# 4. 训练 + 推理 + late fusion
##############################################################################

def main():

    texts, images, labels = load_text_and_image(
        txt_dir="./src/data/",
        img_dir="./src/data/",
        train_file="./src/train.txt",
        img_height=224,
        img_width=224
    )
    print("文本条数:", len(texts))
    print("图像形状:", images.shape)
    print("标签形状:", labels.shape)

    X_text_train, X_text_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        texts, images, labels, test_size=0.1, random_state=42, shuffle=True
    )
    print("训练集大小:", len(X_text_train))
    print("验证集大小:", len(X_text_val))

    num_classes = 3
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

    text_model, text_tokenizer = build_pretrained_text_model(
        model_name="bert-base-uncased",
        num_classes=num_classes
    )
    text_model.summary()

    image_model = build_pretrained_image_model(
        num_classes=num_classes,
        img_height=224,
        img_width=224
    )
    image_model.summary()

    def encode_texts_for_bert(text_list, tokenizer, max_length=64):
        """
        使用 BERT tokenizer 将 list[str] 的文本转成 (input_ids, attention_mask)。
        返回字典: {"input_ids":..., "attention_mask":...}
        """
        all_encoded = tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )
        return {
            "input_ids": all_encoded["input_ids"],
            "attention_mask": all_encoded["attention_mask"]
        }

    train_text_encoded = encode_texts_for_bert(X_text_train, text_tokenizer, max_length=64)
    val_text_encoded = encode_texts_for_bert(X_text_val, text_tokenizer, max_length=64)

    print("\n=== 训练文本模型(BERT) ===")
    text_model.fit(
        train_text_encoded,
        y_train_cat,
        validation_data=(val_text_encoded, y_val_cat),
        epochs=3,
        batch_size=8
    )

    print("\n=== 训练图像模型(ResNet) ===")

    image_model.fit(
        X_img_train,
        y_train_cat,
        validation_data=(X_img_val, y_val_cat),
        epochs=3,
        batch_size=8
    )

    text_probs = text_model.predict(val_text_encoded)
    image_probs = image_model.predict(X_img_val)

    ensemble_probs = (text_probs + image_probs) / 2.0
    final_pred = np.argmax(ensemble_probs, axis=1)

    correct = np.sum(final_pred == y_val)
    total = len(y_val)
    acc = correct / total
    print(f"\n融合后在验证集上的准确率: {acc:.4f}")

if __name__ == "__main__":
    main()
