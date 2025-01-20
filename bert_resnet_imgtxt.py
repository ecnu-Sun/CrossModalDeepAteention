import os
import numpy as np
import pandas as pd
import tensorflow as tf

# 如果要用 Hugging Face 的 transformers
from transformers import BertTokenizer, TFBertModel
# 如果用 GPU，请确保安装的 tensorflow 版本能正确识别 GPU

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
            print(f"[Warning] 文本文件不存在: {txt_path}")
            continue
        if not os.path.isfile(img_path):
            print(f"[Warning] 图片文件不存在: {img_path}")
            continue

        # 读取文本
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text_content = f.read().strip()
        texts.append(text_content)

        # 读取并预处理图像
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
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = TFBertModel.from_pretrained(model_name)

    # 冻结或解冻 BERT
    bert_model.trainable = False

    # 构建 Keras 模型: 输入: dict{"input_ids","attention_mask"} -> BERT -> Dense
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # BERT 输出: last_hidden_state & pooler_output
    # last_hidden_state shape=(batch, seq_len, hidden_size)
    # pooler_output shape=(batch, hidden_size) 通常对应 [CLS] 的表示
    outputs = bert_model([input_ids, attention_mask])
    pooled_output = outputs.pooler_output   # shape=(batch,768) for base model

    # 接一个简单的全连接层做三分类
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

    # 预训练模型
    base_model = ResNet50(
        include_top=False,  # 不要原本的 FC 层
        weights='imagenet',
        pooling='avg',      # GlobalAveragePooling
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  # 冻结主干

    # 自定义顶层
    inputs = Input(shape=(img_height, img_width, 3))
    # 先用官方的预处理
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    # x 现在是特征向量 shape=(None, 2048)
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
    # === (A) 读取文本和图像数据 ===
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

    # === (B) 训练集/验证集划分 ===
    X_text_train, X_text_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        texts, images, labels, test_size=0.1, random_state=42, shuffle=True
    )
    print("训练集大小:", len(X_text_train))
    print("验证集大小:", len(X_text_val))

    # 把标签转换为 one-hot，适用于分类
    num_classes = 3
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

    # === (C1) 构建并编译 文本模型(BERT) ===
    text_model, text_tokenizer = build_pretrained_text_model(
        model_name="bert-base-uncased",
        num_classes=num_classes
    )
    text_model.summary()

    # === (C2) 构建并编译 图像模型(ResNet) ===
    image_model = build_pretrained_image_model(
        num_classes=num_classes,
        img_height=224,
        img_width=224
    )
    image_model.summary()

    # === (D) 把文本转换成 BERT 所需的输入格式 (input_ids, attention_mask) ===
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

    # === (E) 训练文本模型 ===
    print("\n=== 训练文本模型(BERT) ===")
    text_model.fit(
        train_text_encoded,
        y_train_cat,
        validation_data=(val_text_encoded, y_val_cat),
        epochs=3,
        batch_size=8
    )

    # === (F) 训练图像模型 ===
    print("\n=== 训练图像模型(ResNet) ===")
    # 注意：images 在 build_pretrained_image_model 里需要先做预处理 preprocess_input
    # 但上面已在模型内部做了预处理，所以这里直接传原始的图像数组即可
    image_model.fit(
        X_img_train,
        y_train_cat,
        validation_data=(X_img_val, y_val_cat),
        epochs=3,
        batch_size=8
    )

    # === (G) 分别预测验证集概率分布 ===
    text_probs = text_model.predict(val_text_encoded)   # shape=(batch,3)
    image_probs = image_model.predict(X_img_val)        # shape=(batch,3)

    # === (H) late fusion: 取平均后 argmax 得到最终预测标签 ===
    ensemble_probs = (text_probs + image_probs) / 2.0
    final_pred = np.argmax(ensemble_probs, axis=1)

    # === (I) 计算融合后的准确率 ===
    correct = np.sum(final_pred == y_val)
    total = len(y_val)
    acc = correct / total
    print(f"\n融合后在验证集上的准确率: {acc:.4f}")

if __name__ == "__main__":
    main()
