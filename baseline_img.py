import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

txt_file = "./src/train.txt"
data_df = pd.read_csv(txt_file, header=1, names=['guid', 'tag'])

tag_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data_df['label'] = data_df['tag'].map(tag_mapping)

images = []
labels = []

img_height, img_width = 224, 224

for idx, row in data_df.iterrows():
    guid = row['guid']
    label = row['label']

    img_path = os.path.join('./src/data', f"{guid}.jpg")
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} 不存在，跳过该样本。")
        continue

    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    images.append(img_array)
    labels.append(label)

images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

num_classes = 3
labels_one_hot = to_categorical(labels, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels_one_hot,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 3
batch_size = 32
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print("Validation Accuracy: {:.2f}%".format(val_acc * 100))
