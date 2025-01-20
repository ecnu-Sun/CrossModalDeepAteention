import os
import random
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(train_txt_path):
    """
    加载训练集的guid和对应的情感标签，并返回DataFrame。
    """
    df = pd.read_csv(train_txt_path, sep=',', header=0, names=['guid', 'label'])
    return df


def rotate_images(data_folder, df, angles=None):
    """
    读取df中的guid对应的图片并旋转指定角度，保存到同一目录
    训练文本标签不变，但图片增加3份
    这个方法没有提升模型性能，因此弃用
    """
    if angles is None:
        angles = [180]
    for _, row in df.iterrows():
        guid = row['guid']
        img_path = os.path.join(data_folder, f"{guid}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as img:
                for angle in angles:
                    rotated_img = img.rotate(angle)
                    new_filename = f"{guid}_rot{angle}.jpg"
                    rotated_img.save(os.path.join(data_folder, new_filename))
        except:
            # 如果图片异常，可跳过
            pass


def create_train_val_split(df, val_ratio=0.1, random_seed=42):
    """
    将训练数据按指定比例划分为训练集和验证集，并返回两个DataFrame。
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df['label']
    )
    return train_df, val_df


if __name__ == "__main__":
    """
    执行:
    1) 先运行 python data_preprocessing.py （本文件）得到 train_split.txt & val_split.txt
    2) 再运行 python train.py 进行训练+网格搜索
    3) 最后运行 python predicate.py 得到预测文件
    """
    data_folder = "./src/data"
    train_txt = "./src/train.txt"

    df = load_data(train_txt)

    #rotate_images(data_folder, df, angles=[90, 180, 270])

    train_df, val_df = create_train_val_split(df, val_ratio=0.1, random_seed=42)

    train_df.to_csv("./src/train_split.txt", sep='\t', header=False, index=False)
    val_df.to_csv("./src/val_split.txt", sep='\t', header=False, index=False)
    print("数据预处理完成，已将train/val划分结果保存。")
