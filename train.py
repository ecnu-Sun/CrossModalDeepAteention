# train.py

import os
from collections import Counter

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from transformers import AutoProcessor
import torch.nn.functional as F
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from model_definition import MultimodalCrossAttentionClassifier,MultimodalClipClassifier,MultimodalClipClassifierWithAttention,MultimodalClipClassifierWithWeight,MultimodalClipClassifierWithInfoNCE


class MultimodalDataset(Dataset):
    """
    用于多模态情感分类的数据集，可以配置是否只用文本或只用图像做消融实验。
    """

    def __init__(self, df, data_folder, processor, use_text=True, use_image=True):
        """
        df: 包含 [guid, label] 等信息
        data_folder: 图片文件夹
        processor: CLIP 专用预处理 (AutoProcessor)
        use_text/use_image: 控制是否使用文本/图像
        """
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.processor = processor
        self.use_text = use_text
        self.use_image = use_image
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = row['guid']
        label_str = row['label'] if 'label' in self.df.columns else 'neutral'
        label_id = self.label_map[label_str]
        text_filename = f"{guid}.txt"
        text_path = os.path.join(self.data_folder, text_filename)
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='ascii', errors='replace') as f:
                text_str = f.read().strip()
        else:
            raise FileNotFoundError(f"Text file {text_filename} not found in {self.data_folder}.")
        if guid == 890:
            print(text_str)
        # 查找对应图片文件，如 "15.jpg"
        img_filename = f"{guid}.jpg"
        img_path = os.path.join(self.data_folder, img_filename)

        image = None
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")

        # 根据配置调用processor
        if self.use_text and self.use_image:
            inputs = self.processor(
                text=[text_str],
                images=[image],
                return_tensors='pt',
                padding=True
            )
        elif self.use_text and not self.use_image:
            inputs = self.processor(
                text=[text_str],
                return_tensors='pt',
                padding=True
            )
            # 不使用图像时补一个占位pixel_values
            inputs['pixel_values'] = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        elif not self.use_text and self.use_image:
            inputs = self.processor(
                images=[image],
                return_tensors='pt'
            )
            # 不使用文本时补足 input_ids 与 attention_mask
            inputs['input_ids'] = torch.zeros((1, 1), dtype=torch.long)
            inputs['attention_mask'] = torch.ones((1, 1), dtype=torch.long)
        else:
            raise ValueError("必须至少使用文本或图像。")

        sample = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': label_id
        }
        return sample


def evaluate(model, dataloader, device):
    """
    在验证集或测试集评估模型，返回 loss 和 acc。
    """
    model.eval()
    losses = []
    all_preds = []
    all_trues = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = logits.argmax(dim=1).cpu().numpy()
            trues = labels.cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(trues)

    avg_loss = np.mean(losses)
    acc = accuracy_score(all_trues, all_preds)
    return avg_loss, acc


def train_model(
        train_file,
        val_file,
        data_folder,
        logs_dir,
        models_dir,
        use_text=True,
        use_image=True,
        seed=42
):
    """
    训练模型并进行网格搜索，保留最优模型到models_dir。
    use_text/use_image控制消融实验；use_data_augmentation控制是否真的使用旋转图片(需要你在预处理中处理)。
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取train/val文件
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['guid', 'label'])
    val_df = pd.read_csv(val_file, sep='\t', header=None, names=['guid', 'label'])
    print('train_df')
    print(train_df.head(10))

    # 如果不使用数据增强，可在之前的数据预处理中跳过 rotate_images()

    # 初始化 processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("./models/clip-vit-base-patch32")

    # 构建 dataset
    train_dataset = MultimodalDataset(train_df, data_folder, processor, use_text, use_image)
    val_dataset = MultimodalDataset(val_df, data_folder, processor, use_text, use_image)

    label_counter = Counter(train_df['label'].tolist())
    print("训练集各类别分布：", label_counter)

    # 我们的标签映射表在 MultimodalDataset 里是 {'positive': 0, 'neutral': 1, 'negative': 2}
    # 可以从 label_counter 里取对应类别数目，得到 count_positive, count_neutral, count_negative.
    # 当然也可以自动适配更多类别，这里以三分类为例。
    count_positive = label_counter['positive']
    count_neutral = label_counter['neutral']
    count_negative = label_counter['negative']

    # 计算每个类别的权重(与数量的反比)
    max_count = max(count_positive, count_neutral, count_negative)
    weight_positive = max_count / count_positive
    weight_neutral = max_count / count_neutral
    weight_negative = max_count / count_negative
    sample_weights = []
    for label_str in train_df['label'].tolist():
        if label_str == 'positive':
            sample_weights.append(weight_positive)
        elif label_str == 'neutral':
            sample_weights.append(weight_neutral)
        elif label_str == 'negative':
            sample_weights.append(weight_negative)

    # 使用 WeightedRandomSampler 来进行过采样
    sampler = WeightedRandomSampler(
        weights=sample_weights,  # 每个样本对应的采样权重
        num_samples=len(sample_weights),  # 一轮 epoch 中的采样总量一般与原训练集大小相同
        replacement=True  # True表示可以重复采样(过采样)
    )

    # 准备 dataloader
    def collate_fn(batch):
        # 默认实现：把同batch的tensor拼起来
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b['input_ids'] for b in batch], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b['attention_mask'] for b in batch], batch_first=True, padding_value=0
        )
        pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        # print(input_ids)
        # print(attention_mask)
        # print(pixel_values)
        # print(labels)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'label': labels
        }

    # 简单网格搜索示例
    learning_rates = [1e-4]
    batch_sizes = [64]
    epochs_list = [5]

    best_acc = 0.0
    best_model_path = None

    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epochs_list:
                train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)

                model = MultimodalCrossAttentionClassifier(
                    pretrained_model_name="./models/clip-vit-base-patch32",
                    hidden_dim=768,
                    num_labels=3
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()

                train_losses = []
                val_losses = []
                val_accs = []

                for epoch in range(ep):
                    model.train()
                    epoch_train_losses = []

                    for batch in tqdm(train_loader, desc=f"LR={lr}, BS={bs}, EP={ep}, EPOCH={epoch + 1}"):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        pixel_values = batch['pixel_values'].to(device)
                        labels = batch['label'].to(device)

                        optimizer.zero_grad()
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values
                        )
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()

                        epoch_train_losses.append(loss.item())

                    avg_train_loss = np.mean(epoch_train_losses)
                    val_loss, val_acc = evaluate(model, val_loader, device)

                    train_losses.append(avg_train_loss)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    # 记录日志
                    log_str = f"[Epoch {epoch + 1}] LR={lr}, BS={bs}, train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    print(log_str)
                    with open(os.path.join(logs_dir, "train.log"), "a", encoding='utf-8') as f:
                        f.write(log_str + "\n")

                    # 保存最优模型
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model_name = f"best_model_lr{lr}_bs{bs}_ep{ep}.pt"
                        best_model_path = os.path.join(models_dir, best_model_name)
                        torch.save(model.state_dict(), best_model_path)

                # 每个config的最终曲线可画在一张图里或只画最优config
                # 此处仅演示简单保存，可自行改进
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Val Loss')
                plt.legend()
                plt.title("Loss Curve")

                plt.subplot(1, 2, 2)
                plt.plot(val_accs, label='Val Acc')
                plt.legend()
                plt.title("Validation Accuracy")
                plot_name = f"curve_lr{lr}_bs{bs}_ep{ep}.png"
                plt.savefig(os.path.join(logs_dir, plot_name))
                plt.close()

    print(f"网格搜索结束，最佳ACC={best_acc:.4f}，最佳模型路径={best_model_path}")


if __name__ == "__main__":
    """
    执行示例:
    1) 先运行 python data_preprocessing.py 得到 train_split.txt & val_split.txt
    2) 再运行 python train.py (可自行修改路径) 进行训练+网格搜索
    """
    data_folder = "./src/data"
    train_file = "./src/train_split.txt"
    val_file = "./src/val_split.txt"
    logs_dir = "./logs"
    models_dir = "./models"

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    train_model(
        train_file=train_file,
        val_file=val_file,
        data_folder=data_folder,
        logs_dir=logs_dir,
        models_dir=models_dir,
        use_text=True,
        use_image=True,
        seed=42
    )
