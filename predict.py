# predict.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from train import MultimodalDataset
from model_definition import MultimodalCrossAttentionClassifier


def predict_test_set(
        test_txt,
        model_path,
        output_path,
        data_folder,
        use_text=True,
        use_image=True
):
    """
    使用训练好的，对test_without_label.txt进行预测，并写回文件。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(test_txt, sep=',', header=0, names=['guid', 'label'])
    print(df.head())
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = MultimodalDataset(df, data_folder, processor, use_text, use_image)

    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b['input_ids'] for b in batch], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b['attention_mask'] for b in batch], batch_first=True, padding_value=0
        )
        pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'label': labels
        }

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    label_map = {0: "positive", 1: "neutral", 2: "negative"}

    model = MultimodalCrossAttentionClassifier(
        pretrained_model_name="openai/clip-vit-base-patch32",
        hidden_dim=768,
        num_labels=3
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            pred = logits.argmax(dim=1).cpu().item()
            all_preds.append(label_map[pred])

    df['label'] = all_preds
    df.to_csv(output_path, sep=',', index=False, header=['guid', 'tag'])
    print(f"预测完成，结果已保存到 {output_path}")


if __name__ == "__main__":
    """
    执行:
    1) 先运行 python data_preprocessing.py 得到 train_split.txt & val_split.txt
    2) 再运行 python train.py 进行训练+网格搜索
    3) 最后运行 python predicate.py（本文件） 得到预测文件
    """
    test_txt = "./src/test_without_label.txt"
    data_folder = "./src/data"
    best_model_path = "./models/best_model_lr3e-06_bs64_ep3.pt"
    output_path = "./src/test_predicted.txt"

    predict_test_set(
        test_txt=test_txt,
        model_path=best_model_path,
        output_path=output_path,
        data_folder=data_folder,
        use_text=True,
        use_image=True
    )
